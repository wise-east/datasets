# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Semantic Scholar's records for research papers published in all fields"""


import json
import re
import os 
from loguru import logger 
import datasets


_CITATION = """\
@misc{lo2020s2orc,
      title={S2ORC: The Semantic Scholar Open Research Corpus},
      author={Kyle Lo and Lucy Lu Wang and Mark Neumann and Rodney Kinney and Dan S. Weld},
      year={2020},
      eprint={1911.02782},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
A large corpus of 81.1M English-language academic papers spanning many academic disciplines.
Rich metadata, paper abstracts, resolved bibliographic references, as well as structured full
text for 8.1M open access papers. Full text annotated with automatically-detected inline mentions of
citations, figures, and tables, each linked to their corresponding paper objects. Aggregated papers
from hundreds of academic publishers and digital archives into a unified source, and create the largest
publicly-available collection of machine-readable academic text to date.
"""
_HOMEPAGE="https://allenai.org/data/s2orc" 


_LICENSE = "Semantic Scholar Open Research Corpus is licensed under ODC-BY."

DATASETS_DIR = os.environ.get("DATASETS_DIR", "")
if not DATASETS_DIR: 
    logger.error("DATASETS_DIR is not set. It needs to be set as an environment variable. Make sure it points to the Hugging Face `datasets/` directory.")
    raise EnvironmentError

VERSION = datasets.Version("1.0.0")

class S2ORC_DAPT_Config(datasets.BuilderConfig):
    """BuilderConfig for SuperGLUE."""

    def __init__(self, datapath, citation, **kwargs):
        """BuilderConfig for S2ORC DAPT.

        Args:
        datapath: filename for the domain of interest
        """
        # Version history:
        # 1.0.2: Fixed non-nondeterminism in ReCoRD.
        # 1.0.1: Change from the pre-release trial version of SuperGLUE (v1.9) to
        #        the full release (v2.0).
        # 1.0.0: S3 (new shuffling, sharding and slicing mechanism).
        # 0.0.2: Initial version.
        super(S2ORC_DAPT_Config, self).__init__(version=VERSION, **kwargs)
        self.datapath = datapath 
        self.citation = citation 
    

# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class S2ORC_DAPT(datasets.GeneratorBasedBuilder):
    """
    Hugging Face Dataset for using S2ORC for Domain adaptation Pre-Training (DAPT)
    """

    BUILDER_CONFIGS = [
        S2ORC_DAPT_Config(
            name="cs", 
            description=_DESCRIPTION,
            citation=_CITATION,
            datapath=os.path.join(DATASETS_DIR,  "datasets/s2orc_ours/s2orc_Computer Science_hf_data_lines.txt")
        ),
        S2ORC_DAPT_Config(
            name="biomed",
            description=_DESCRIPTION,
            citation=_CITATION,
            datapath=os.path.join(DATASETS_DIR, "datasets/s2orc_ours/s2orc_Biology_Medicine_hf_data_lines.txt")
        )
    ]
    DEFAULT_CONFIG_NAME = "cs"

    def _info(self):

        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                # "paper_id": datasets.Value("string"),
                "text": datasets.Value("string"),
                # "abstract": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """
        Preparing the data should be done in the parent directory with `filter_s2orc.py` 
        """

        # import pdb; pdb.set_trace() 

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": self.config.datapath,
                    "split": "train",
                }
            )
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                text = row.replace("\n", "") 
                yield id_, {
                    "text": text,
                    "id": id_
                }

    def _generate_examples_jsonl(self, filepath, split):
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                yield id_, {
                    "abstract": data["abstract"],
                    "text": data["body_text"],
                    "paper_id": data["paper_id"],
                    "id": id_
                }