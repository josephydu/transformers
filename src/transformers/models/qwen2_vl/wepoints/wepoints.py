import math
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import (
    GenerationMixin,
    PreTrainedModel,
    PreTrainedTokenizer,
    Qwen2ForCausalLM,
)
import torch.nn.functional as F

try:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (  # noqa
        Qwen2VLImageProcessor,
    )
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
    from transformers.models.qwen2_vl.modeling_qwen2_vl import PatchMerger
except ImportError:
    print('Please upgrade transformers to version 4.46.3 or higher')

from .configuration_pointsv15_chat import POINTSV15ChatConfig


class Qwen2VisionTransformerForNavitPOINTS(
        Qwen2VisionTransformerPretrainedModel):  # noqa
    """Rewrite the forward function of Qwen2VisionTransformerPretrainedModel to
    adapt to POINTS.  # noqa.

    Do no apply patch merging to the hidden features output by the transformer.
    """
    def __init__(self, config) -> None:
        super().__init__(config)

    def forward(self, hidden_states: torch.Tensor,
                grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                             grid_thw[:, 0]).cumsum(
                                                 dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(hidden_states,
                                cu_seqlens=cu_seqlens,
                                rotary_pos_emb=rotary_pos_emb)

        return hidden_states



class POINTSV15ChatModel(PreTrainedModel, GenerationMixin):
    config_class = POINTSV15ChatConfig
    _no_split_modules = ["CustomLlamaLayer",
                         "Qwen2VisionTransformerPretrainedModel"]

    """Chat model for POINTSv1.5.
    
    Args:
        config (POINTSChatConfigV15): The model config.
    """

    def __init__(self, config: POINTSV15ChatConfig) -> None:
        super().__init__(config)
        self.llm = Qwen2ForCausalLM(config.llm_config)
        self.vision_encoder = Qwen2VisionTransformerForNavitPOINTS._from_config(  # noqa
            config.vision_config, attn_implementation="flash_attention_2"
        )
        self.vision_projector = PatchMerger(config.llm_config.hidden_size,
                                            context_dim=1280)

    def process_images(self, images: torch.Tensor, 
                       image_grid_thws: List[list]) -> torch.Tensor:
        """Obtain image features from the vision encoder.
        
        Args:
            images (torch.Tensor): The input images.
            image_grid_thws (List[list]): The grid thresholds for the images.

        Returns:
            torch.Tensor: The image features.
        """
        image_features = self.vision_encoder(images, grid_thw=image_grid_thws)
        image_features = self.vision_projector(image_features)
        return image_features
    
    def construct_prompt(self, messages: List[dict],
                         image_processor: Qwen2VLImageProcessor) -> Tuple[str, List[Image.Image], List[list]]: # noqa
        """Construct the prompt for the chat model.

        Args:
            messages (List[dict]): The input messages.
        
        Returns:
            Tuple[str, List[Image.Image], List[list]]: 
                The prompt, images, and image grid shape.
        """
        images = []
        image_grid_thws = []
        reconstructed_messages = []
        for message in messages:
            role = message['role']
            content_from_role = ''
            for item in message['content']:
                if item['type'] == 'text':
                    content_from_role += item['text']
                elif item['type'] == 'image':
                    image_path = item['image']
                    max_pixels = item['max_pixels'] if 'max_pixels' in item else None
                    image = Image.open(image_path).convert('RGB')
                    if max_pixels is not None:
                        # obtain image size
                        width, height = image.size
                        cur_image_pixels = width * height
                        if cur_image_pixels > max_pixels:
                            beta = math.sqrt((height * width) / max_pixels)
                            new_width = math.floor(width / beta)
                            new_height = math.floor(height / beta)
                            image = image.resize((new_width, new_height))
                    image_data = image_processor(images=image)
                    pixel_values = image_data['pixel_values']
                    image_grid_thw = image_data['image_grid_thw']
                    images.extend(pixel_values)
                    image_grid_thws.append(image_grid_thw)
                    seq_len = int(image_grid_thw[0][1] * image_grid_thw[0][2] / 4) # noqa
                    content_from_role += '<|vision_start|>' + '<|image_pad|>' * seq_len + '<|vision_end|>' + '\n' # noqa
            reconstructed_messages.append({
                'role': role,
                'content': content_from_role
            })
        prompt = self.apply_chat_template(reconstructed_messages)
        return prompt, images, image_grid_thws
    
    def apply_chat_template(self, messages: List[dict]) -> str:
        """Apply the chat template to the input messages.

        Args:
            messages (List[dict]): The input messages.
        
        Returns:
            str: The prompt.
        """
        role_prefix_mapping = {
            'user': '<|im_start|>user\n',
            'assistant': '<|im_start|>assistant\n'
        }
        role = 'user'
        prompt = ''
        for message in messages:
            role = message['role']
            content = message['content']
            prompt += role_prefix_mapping[role] + content + '<|im_end|>\n'
        if role == 'user':
            prompt += '<|im_start|>assistant\n'
        return prompt

    @torch.no_grad()
    def chat(self, 
             messages: List[dict],
             tokenizer: PreTrainedTokenizer,
             image_processor: object,
             generation_config: dict = None) -> str:
        """Generate a response to the input prompt.

        Args:
            messages (List[dict]): The input messages.
            tokenizer (PreTrainedTokenizer): The tokenizer to use.
            image_processor (object): The image processor to use.
            generation_config (dict, optional): The generation config. 
                Defaults to None.
        Returns:
            str: The generated response.
        """
        prompt, images, image_grid_thws = self.construct_prompt(
            messages, image_processor
        )
        images = np.array(images)
        images = torch.from_numpy(images).to(self.vision_encoder.device).to(self.vision_encoder.dtype) # noqa
        image_grid_thws = np.concatenate(image_grid_thws, axis=0)
        image_grid_thws = (
            torch.from_numpy(image_grid_thws)
            .cuda()
            .long()
        )
        image_features = self.vision_encoder(images, grid_thw=image_grid_thws)
        image_features = self.vision_projector(image_features)
        model_inputs = tokenizer(prompt, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        # stop token
        eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        # image token
        image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        generation_config.update(
            {
                'eos_token_id': eos_token_id,
            }
        )
        outputs = self.generate(
            input_ids=input_ids,
            image_grid_thws=image_grid_thws,
            attention_mask=attention_mask,
            image_features=[image_features],
            image_token_id=image_token_id,
            **generation_config
        )
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return response
      
    def _split_input_ids(self, input_ids, special_token):
        special_pos = input_ids == special_token
        pos = (special_pos[:-1] != special_pos[1:]).nonzero() + 1
        if pos.shape[0] % 2 != 0:
            pos = torch.cat([torch.tensor([[0]]).to(pos.device), pos])
        pos = pos.reshape(-1, 2).tolist()
        return pos

    def generate(self,
                 input_ids: torch.LongTensor,
                 image_grid_thws: torch.LongTensor,
                 attention_mask: torch.LongTensor,
                 image_features: List[torch.Tensor],
                 image_token_id: int,
                 generation_config: Optional[dict] = None,
                 output_hidden_states: Optional[bool] = None,
                 **generate_kwargs) -> torch.LongTensor:
        input_embeddings = self.llm.model.embed_tokens(input_ids)
        batch_size = input_ids.shape[0]
        assert len(image_features) == batch_size
        for i in range(batch_size):
            pos = self._split_input_ids(input_ids[i], image_token_id)
            assert len(pos) == len(image_grid_thws)
            image_pos = [
                int(image_grid_thw[1] * image_grid_thw[2] / 4)
                for image_grid_thw in image_grid_thws
            ]
            image_pos.insert(0, 0)
            image_pos = np.cumsum(image_pos)
            for j, (start, end) in enumerate(pos):
                input_embeddings[i, start:end] = \
                    image_features[i][image_pos[j]:image_pos[j+1]]
        outputs = self.llm.generate(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs
        )
        return outputs
