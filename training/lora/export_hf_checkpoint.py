import fire
import torch
import transformers
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def main(
    base_model: str = "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
    adapter_path: str = "outputs/redpajama-incite-chat-3b-lowrank",
    output_path: str = "outputs/redpajama-incite-chat-3b-lowrank-hf",
):
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )

    first_weight = base_model.model.layers[0].attention.query_key_value.weight
    first_weight_old = first_weight.clone()

    lora_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )

    lora_weight = lora_model.base_model.model.gpt_neox.layers[
        0
    ].attention.query_key_value.weight

    assert torch.allclose(first_weight_old, first_weight)

    # merge weights
    lora_model = lora_model.merge_and_unload()

    lora_model.train(False)

    # did we do anything?
    assert not torch.allclose(first_weight_old, first_weight)

    lora_model_state_dict = lora_model.state_dict()
    state_dict = {
        k.replace("base_model.model.", ""): v
        for k, v in lora_model_state_dict.items()
        if "lora" not in k
    }

    base_model.save_pretrained(
        output_path, state_dict=state_dict, max_shard_size="400MB"
    )
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    fire.Fire(main)
