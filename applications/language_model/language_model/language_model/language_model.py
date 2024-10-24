from _utilities.cuda import detect_cuda
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from language_model.language_model.template import get_chat_template
import time
import logging

logger = logging.getLogger(__name__)

# TODO: Add tools support
#  https://docs.mistral.ai/capabilities/function_calling/
#  https://huggingface.co/docs/transformers/main/chat_templating#introduction
# https://medium.com/@yufan1602/modular-rag-and-rag-flow-part-%E2%85%B0-e69b32dc13a3
# https://medium.com/@yufan1602/modular-rag-and-rag-flow-part-ii-77b62bf8a5d3
# https://towardsdatascience.com/implementing-modular-rag-with-haystack-and-hypster-d2f0ecc88b8f


class LanguageModel:
    def __init__(self, model_path=None, device_map="cpu"):
        logger.debug(f"CUDA Available: {torch.cuda.is_available()}")
        logger.debug("Create tokenizer")
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path, device_map=device_map
        )
        logger.debug("Tokenizer created")
        model_start = time.time()
        logger.debug("Starting model")
        # TODO: Fix Some weights of the model checkpoint (...) were not used when initializing MistralForCausalLM

        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            device_map=device_map,
            attn_implementation="eager",
        )
        logger.debug(
            f"Model started in {(time.time()-model_start):0.3f}s, {self._model.device}"
        )
        self._model.generation_config.pad_token_id = self._tokenizer.eos_token_id
        self._model.generation_config.eos_token_id = self._tokenizer.eos_token_id

    def prompt(
        self,
        conversation,
        token_generated_callback: lambda x: None = None,
        response_generated_callback: lambda x: None = None,
        max_length=1000,
        temperature=0.3,
    ):
        def prompt_tokenizer(data):
            return self._tokenizer.encode_plus(
                text=data,
                return_tensors="pt",
            )

        return self._generate(
            conversation=conversation,
            token_generated_callback=token_generated_callback,
            response_generated_callback=response_generated_callback,
            max_length=max_length,
            temperature=temperature,
            tokenizer=prompt_tokenizer,
        )

    def generate(
        self,
        conversation,
        token_generated_callback: lambda x: None = None,
        response_generated_callback: lambda x: None = None,
        max_length=1000,
        temperature=0.3,
    ):
        def conversation_tokenizer(data):
            return self._tokenizer.apply_chat_template(
                conversation=data,
                chat_template=get_chat_template(),
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

        return self._generate(
            conversation=conversation,
            token_generated_callback=token_generated_callback,
            response_generated_callback=response_generated_callback,
            max_length=max_length,
            temperature=temperature,
            tokenizer=conversation_tokenizer,
        )

    def _generate(
        self,
        conversation,
        token_generated_callback: lambda x: None = None,
        response_generated_callback: lambda x: None = None,
        max_length=1000,
        temperature=0.3,
        tokenizer=None,
    ):
        eos_token_id = self._model.config.eos_token_id

        generate_start = time.time()
        logger.debug("Infering response")
        tokens = tokenizer(conversation)

        input_ids = tokens["input_ids"].to(self._model.device)
        # print("PROMPT", self._tokenizer.decode(input_ids[0]))
        attention_mask = tokens["attention_mask"].to(self._model.device)
        logger.debug(f"Request tokenized in {time.time()-generate_start:0.3f}s")

        # outputs = self._model.generate(
        #                 input_ids,
        #                 max_new_tokens=1000,
        #                 attention_mask=attention_mask,
        #                 temperature=temperature,
        #                 do_sample=True,)

        # logger.debug(f"Response inferred in {time.time()-generate_start:0.3f}s")
        # input_length = input_ids.size(1)
        # # print(outputs)
        # return self._tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        result = []
        for i in range(max_length):
            outputs = self._model.generate(
                input_ids,
                max_new_tokens=1,
                attention_mask=attention_mask,
                temperature=temperature,
                do_sample=True,
            )
            if i == 0:
                logger.debug(
                    f"Time to first token: {time.time() - generate_start:0.3f} seconds"
                )

            input_length = input_ids.size(1)

            new_tokens = outputs[:, input_length:]

            input_ids = torch.cat([input_ids, new_tokens], dim=-1)
            # Update attention mask to include the new token (mark it as "non-padded")
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], new_tokens.size(1)),
                        device=attention_mask.device,
                    ),
                ],
                dim=-1,
            )

            if new_tokens.item() == eos_token_id:
                break

            decoded_word = self._decode_single_word(outputs, first_word=i == 0)
            # print(decoded_word, has_space)
            # if has_space:
            #     result.append(" ")
            result.append(decoded_word)
            if token_generated_callback:
                token_generated_callback(decoded_word)
            # logger.debug("%d: [%s]", i, f"{' ' if has_space else ''}{decoded_word}")

        logger.debug(f"Response inferred in {time.time()-generate_start:0.3f}s")
        joined_result = "".join(result)
        if response_generated_callback is not None:
            response_generated_callback(joined_result)
        return joined_result

    def _decode_single_word(self, outputs, first_word=False):
        decoded_word = self._tokenizer.convert_ids_to_tokens(
            outputs[0], skip_special_tokens=True
        )[-1]
        return decoded_word.replace("Ċ", "\n").replace("Ġ", " ")
