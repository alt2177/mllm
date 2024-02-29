"""
Class file for the actual MLLM we want to use
"""

import torch
import yaml
import mergekit as mk
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge
import os





class MLLM:


    def __init__(self) -> None:
        """
        constructor
        """
        # variables necessary for mergekit
        OUTPUT_PATH = "./merged"  # folder to store the result in
        LORA_MERGE_CACHE = "/tmp"  # change if you want to keep these for some reason
        CONFIG_YML = "../tests/ultra_llm_merged.yml"  # merge configuration file
        COPY_TOKENIZER = True  # you want a tokenizer? yeah, that's what i thought
        LAZY_UNPICKLE = False  # experimental low-memory model loader
        LOW_CPU_MEMORY = False  # enable if you somehow have more VRAM than RAM+swap

        # self.set_paths()
        with open(CONFIG_YML, "r", encoding="utf-8") as fp:
            merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))
        run_merge(
             merge_config,
             out_path=OUTPUT_PATH,
             options=MergeOptions(
                 lora_merge_cache=LORA_MERGE_CACHE,
                 cuda=torch.cuda.is_available(),
                 copy_tokenizer=COPY_TOKENIZER,
                 lazy_unpickle=LAZY_UNPICKLE,
                 low_cpu_memory=LOW_CPU_MEMORY,
             ),
         )
        print("Done!")

        pass

    def set_paths(self) -> None:
        """
        Make sure we have the right path to access yaml files
        """
        # create our full path
        full_path = os.path.join(os.getcwd(), "mergekit-yaml")

        # set path we want to change to
        new_directory = "venv/bin/"

        # check if we have mergekit in our cwd
        if not os.path.exists(full_path):
            try:
                os.chdir(new_directory)
                print(f"Directory changed to: {os.getcwd()}")
            except FileNotFoundError:
                print(f"Directory not found: {new_directory}")
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            print(f"{file_or_directory} exists in the current directory. No change made.")
        print(os.getcwd())
        pass




