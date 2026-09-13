---
license: cc-by-sa-4.0
task_categories:
- question-answering
language:
- en
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
dataset_info:
  features:
  - name: user_prompt_index
    dtype: int64
  - name: user_prompt
    dtype: string
  - name: answerable
    dtype: bool
  - name: information_type
    dtype: string
  - name: category
    dtype: string
  - name: tags
    sequence: string
  - name: information_date
    dtype: string
  - name: documents
    list:
    - name: text
      dtype: string
    - name: title
      dtype: string
  - name: documents_str
    dtype: string
  - name: rag_prompt
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: responses
    struct:
    - name: Llama-2-7b-chat-hf
      struct:
      - name: details
        struct:
        - name: annotations
          struct:
          - name: original_output
            struct:
            - name: addressed_user_request
              dtype: bool
            - name: addressed_user_request_analysis
              dtype: string
            - name: cluelessness
              dtype: bool
            - name: cluelessness_analysis
              dtype: string
            - name: completely_hallucinated
              dtype: bool
            - name: hallucination_as_a_whole_analysis
              dtype: string
            - name: hallucinations
              list:
              - name: final_substring
                dtype: string
              - name: substring_analysis
                dtype: string
              - name: substring_with_context
                dtype: string
            - name: hallucinations_analysis
              dtype: string
            - name: partially_hallucinated
              dtype: bool
          - name: result
            struct:
            - name: addressed_user_prompt
              dtype: bool
            - name: all_valid
              dtype: bool
            - name: cluelessness
              dtype: bool
            - name: completely_hallucinated
              dtype: bool
            - name: hallucinations
              list:
              - name: end
                dtype: int64
              - name: start
                dtype: int64
              - name: text
                dtype: string
              - name: valid
                dtype: bool
        - name: answerable
          dtype: bool
        - name: full_chat
          dtype: string
        - name: full_prompt
          dtype: string
        - name: group
          dtype: string
        - name: output
          dtype: string
        - name: token_starts
          sequence: int64
      - name: hallucinations
        list:
        - name: end
          dtype: int64
        - name: start
          dtype: int64
        - name: text
          dtype: string
        - name: valid
          dtype: bool
      - name: text
        dtype: string
    - name: Llama-3.1-8B-Instruct
      struct:
      - name: details
        struct:
        - name: annotations
          struct:
          - name: original_output
            struct:
            - name: addressed_user_request
              dtype: bool
            - name: addressed_user_request_analysis
              dtype: string
            - name: cluelessness
              dtype: bool
            - name: cluelessness_analysis
              dtype: string
            - name: completely_hallucinated
              dtype: bool
            - name: hallucination_as_a_whole_analysis
              dtype: string
            - name: hallucinations
              list:
              - name: final_substring
                dtype: string
              - name: substring_analysis
                dtype: string
              - name: substring_with_context
                dtype: string
            - name: hallucinations_analysis
              dtype: string
            - name: partially_hallucinated
              dtype: bool
          - name: result
            struct:
            - name: addressed_user_prompt
              dtype: bool
            - name: all_valid
              dtype: bool
            - name: cluelessness
              dtype: bool
            - name: completely_hallucinated
              dtype: bool
            - name: hallucinations
              list:
              - name: end
                dtype: int64
              - name: start
                dtype: int64
              - name: text
                dtype: string
              - name: valid
                dtype: bool
        - name: answerable
          dtype: bool
        - name: full_chat
          dtype: string
        - name: full_prompt
          dtype: string
        - name: group
          dtype: string
        - name: output
          dtype: string
        - name: token_starts
          sequence: int64
      - name: hallucinations
        list:
        - name: end
          dtype: int64
        - name: start
          dtype: int64
        - name: text
          dtype: string
        - name: valid
          dtype: bool
      - name: text
        dtype: string
    - name: Mistral-7B-Instruct-v0.1
      struct:
      - name: details
        struct:
        - name: annotations
          struct:
          - name: original_output
            struct:
            - name: addressed_user_request
              dtype: bool
            - name: addressed_user_request_analysis
              dtype: string
            - name: cluelessness
              dtype: bool
            - name: cluelessness_analysis
              dtype: string
            - name: completely_hallucinated
              dtype: bool
            - name: hallucination_as_a_whole_analysis
              dtype: string
            - name: hallucinations
              list:
              - name: final_substring
                dtype: string
              - name: substring_analysis
                dtype: string
              - name: substring_with_context
                dtype: string
            - name: hallucinations_analysis
              dtype: string
            - name: partially_hallucinated
              dtype: bool
          - name: result
            struct:
            - name: addressed_user_prompt
              dtype: bool
            - name: all_valid
              dtype: bool
            - name: cluelessness
              dtype: bool
            - name: completely_hallucinated
              dtype: bool
            - name: hallucinations
              list:
              - name: end
                dtype: int64
              - name: start
                dtype: int64
              - name: text
                dtype: string
              - name: valid
                dtype: bool
        - name: answerable
          dtype: bool
        - name: full_chat
          dtype: string
        - name: full_prompt
          dtype: string
        - name: group
          dtype: string
        - name: output
          dtype: string
        - name: token_starts
          sequence: int64
      - name: hallucinations
        list:
        - name: end
          dtype: int64
        - name: start
          dtype: int64
        - name: text
          dtype: string
        - name: valid
          dtype: bool
      - name: text
        dtype: string
    - name: Mistral-7B-Instruct-v0.3
      struct:
      - name: details
        struct:
        - name: annotations
          struct:
          - name: original_output
            struct:
            - name: addressed_user_request
              dtype: bool
            - name: addressed_user_request_analysis
              dtype: string
            - name: cluelessness
              dtype: bool
            - name: cluelessness_analysis
              dtype: string
            - name: completely_hallucinated
              dtype: bool
            - name: hallucination_as_a_whole_analysis
              dtype: string
            - name: hallucinations
              list:
              - name: final_substring
                dtype: string
              - name: substring_analysis
                dtype: string
              - name: substring_with_context
                dtype: string
            - name: hallucinations_analysis
              dtype: string
            - name: partially_hallucinated
              dtype: bool
          - name: result
            struct:
            - name: addressed_user_prompt
              dtype: bool
            - name: all_valid
              dtype: bool
            - name: cluelessness
              dtype: bool
            - name: completely_hallucinated
              dtype: bool
            - name: hallucinations
              list:
              - name: end
                dtype: int64
              - name: start
                dtype: int64
              - name: text
                dtype: string
              - name: valid
                dtype: bool
        - name: answerable
          dtype: bool
        - name: full_chat
          dtype: string
        - name: full_prompt
          dtype: string
        - name: group
          dtype: string
        - name: output
          dtype: string
        - name: token_starts
          sequence: int64
      - name: hallucinations
        list:
        - name: end
          dtype: int64
        - name: start
          dtype: int64
        - name: text
          dtype: string
        - name: valid
          dtype: bool
      - name: text
        dtype: string
    - name: golden_answer
      dtype: string
  - name: details
    struct:
    - name: amount_of_chunks
      dtype: int64
    - name: answerable_chunks
      list:
      - name: passage_index
        dtype: int64
      - name: suitable_article_index
        dtype: int64
      - name: suitable_article_title
        dtype: string
      - name: text
        dtype: string
    - name: info_answerable
      dtype: string
    - name: info_unanswerable
      dtype: string
    - name: source_chunk_index
      dtype: int64
    - name: template
      list:
      - name: content
        dtype: string
      - name: role
        dtype: string
    - name: template_details
      struct:
      - name: sample_log
        struct:
        - name: answer_suffix
          dtype: bool
        - name: info_label
          struct:
          - name: plural
            dtype: string
          - name: singular
            dtype: string
        - name: info_prov
          dtype: string
        - name: instr_before_info
          dtype: bool
        - name: instructions
          dtype: string
        - name: md_capitalize
          dtype: bool
        - name: md_hashtags
          dtype: string
        - name: md_info_prov_title
          dtype: string
        - name: md_instr_title
          dtype: string
        - name: prompt_format
          dtype: string
        - name: question_phrase
          dtype: string
        - name: section_emb
          dtype: string
        - name: system_message
          dtype: string
        - name: unansw_hint
          dtype: string
      - name: settings
        struct:
        - name: answer_suffix
          dtype: bool
        - name: has_sys_msg
          dtype: bool
        - name: info_in_sys_msg
          dtype: bool
        - name: instructions_in_sys_msg
          dtype: bool
        - name: question_after_info
          dtype: bool
        - name: unanswerability_hint
          dtype: bool
        - name: unanswerability_hint_in_sys_msg
          dtype: bool
    - name: unanswerable_chunks
      list:
      - name: passage_index
        dtype: int64
      - name: suitable_article_index
        dtype: int64
      - name: suitable_article_title
        dtype: string
      - name: text
        dtype: string
    - name: user_prompt
      struct:
      - name: article_title
        dtype: string
      - name: category
        dtype: string
      - name: cluelessness_answer
        dtype: string
      - name: details
        struct:
        - name: answer_quote_from_passage
          dtype: string
        - name: passage_index
          dtype: int64
        - name: suitable_article
          struct:
          - name: backlinks
            dtype: int64
          - name: content
            dtype: string
          - name: created_en
            dtype: string
          - name: earliest_creation_date
            dtype: string
          - name: passage_data
            list:
            - name: contains_article
              dtype: bool
            - name: earliest_access_date
              dtype: string
            - name: earliest_archive_date
              dtype: string
            - name: earliest_date
              dtype: string
            - name: end
              dtype: int64
            - name: references
              list:
              - name: access_date
                dtype: string
              - name: archive_date
                dtype: string
              - name: date
                dtype: string
              - name: key_count
                dtype: int64
              - name: ref_label
                dtype: string
            - name: start
              dtype: int64
          - name: removed_duplicates
            dtype: bool
          - name: retrieval_date_utc
            dtype: string
          - name: revision_id
            dtype: int64
          - name: title
            dtype: string
          - name: url
            dtype: string
        - name: suitable_article_index
          dtype: int64
        - name: user_prompt_requirements
          sequence: string
      - name: golden_answer
        dtype: string
      - name: information_type
        dtype: string
      - name: passage_containing_answer
        dtype: string
      - name: user_prompt
        dtype: string
      - name: user_prompt_index
        dtype: int64
  splits:
  - name: train
    num_bytes: 85984236
    num_examples: 1842
  - name: test
    num_bytes: 132280358
    num_examples: 2781
  download_size: 76699735
  dataset_size: 218264594
---

# RAGognize Dataset Card

| Resource      | Link |
|---------------|------|
| **Code**      | [![GitHub Repo](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/F4biian/RAGognizer) |
| **Paper**     | [![arXiv Paper](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://arxiv.org/abs/2604.15945v1) |
| **Demo**     | [![🤗 Space](https://img.shields.io/badge/🤗%20Space-Demo-yellow)](https://huggingface.co/spaces/F4biian/RAGognizer) |

In Retrieval-Augmented Generation (RAG), ensuring accuracy and reliability is essential. **RAGognize** is a dataset created to help researchers and developers study and improve how AI systems use retrieved information. By offering structured examples and **natural** token-level closed-domain hallucination annotations, it provides a resource for analyzing model behavior and developing methods that might help make RAG outputs more factual and dependable.

**RAGognize** is an English-only Wikipedia-based dataset designed to support a wide range of tasks:

1) **Benchmarking Hallucination Detection:** Test your detection models against a diverse set of outputs. Each prompt is paired with a "golden" answer from `gemini-2.5-Pro` and responses from `Llama-2-7b-chat-hf`, `Llama-3.1-8B-Instruct`, `Mistral-7B-Instruct-v0.1`, and `Mistral-7B-Instruct-v0.3`, all annotated for hallucinations at the token level. (**Note: All hallucnations occured naturally, i.e. were definitely produced by the corresponding LLM!**)

2) **Supervised Finetuning:** Leverage the "golden" answers to train your models to be more discerning. Fine-tuning on **RAGognize** may help models learn to ground their responses in provided context.

3) **Head-Injected Finetuning:** The granular, token-level annotations can be used to finetune an LLM on its original inputs, but also an injected MLP at a certain layer of that LLM. Finetuning that newly added MLP and LLM simultaneously helps separating hallucinatory internal states from non-hallucinatory, making it ideal for hallucination detectors. This is exactly what we did, resulting in our detector, **RAGognizer**. For more details, see our paper: (Coming Soon)

4) **Further Analyses:** Analyze how different properties of RAG prompts and user prompts affect model responses. **RAGognize’s** detailed metadata provides a good foundation for analyzing what properties might drive hallucinations or other unintended behaviour. 

> Every information that is requested by the user prompt should have emerged after the cutoff-date **"2024-05-23"**, which has been selected to ensure that all information are unlikely to persist in the parametric knowledge of the four models used (`Llama-2-7b-chat-hf`, `Llama-3.1-8B-Instruct`, `Mistral-7B-Instruct-v0.1`, and `Mistral-7B-Instruct-v0.3`). This made it much easier to consider a RAG prompt unanswerable if the particular information is not part of it.

> The test split of RAGognize also contains five non-annotated responses per LLM, samped at a temperature of 0.7. This is separately available at [**RAGognize-with-samples-test**](https://huggingface.co/datasets/F4biian/RAGognize-with-samples-test).

## Dataset Structure

Each entry in the dataset is a JSON object with the following fields:

*   `user_prompt_index`: (int) A unique identifier for the user prompt.
*   `user_prompt`: (str) The user's question.
*   `answerable`: (bool) A boolean flag indicating whether the provided documents contain the answer to the `user_prompt`.
*   `information_type`: (str) The type of information the user is seeking (e.g., "NUMERICAL", "TEMPORAL").
*   `category`: (str) The general topic of the user's prompt (e.g., "SPORTS", "POLITICS").
*   `tags`: (list of str) A list of tags describing the characteristics of the user prompt (e.g., "QUESTION", "FLAWED").
*   `information_date`: (str) The point in time when the information being requested should have emerged.
*   `documents`: (list of dict) A list of retrieved documents. Each document is a dictionary with:
    *   `title`: (str) The title of the document.
    *   `text`: (str) The content of the document.
*   `documents_str`: (str) A formatted string containing the content of all retrieved documents.
*   `rag_prompt`: (list of dict) The complete prompt for the language model, formatted for a conversational AI. It includes the user message and sometimes the system message. All possible roles are "user" or "system".
*   `responses`: (dict) A dictionary containing the "golden" answer and responses from various large language models. This includes:
    *   `golden_answer`: (str) The ideal, correct answer to the user's prompt based on the provided documents. This has been generated by `gemini-2.5-pro`.
    *   Model-specific responses (`Llama-2-7b-chat-hf`, `Llama-3.1-8B-Instruct`, `Mistral-7B-Instruct-v0.1`, and `Mistral-7B-Instruct-v0.3`): Each of these is a dictionary containing:
        *   `text`: (str) The response generated by the modelat temperature `0.0`.
        *   `hallucinations`: (list) A list of any hallucinated content in the model's response.
        *   `details`: (dict) Further details about the model's response, including the full prompt and chat history.
*   `details`: (dict) A dictionary containing metadata about the data generation process, including information about the source chunks, templates used, and details about the original user prompt generation.

## Stats

### Answerability
| answerable    | train       | test         | total        |
|:--------------|------------:|-------------:|-------------:|
| True  | 924 (50.2%) | 1391 (50.0%) | 2315 (50.1%) |
| False | 918 (49.8%) | 1390 (50.0%) | 2308 (49.9%) |
| **total**         | 1842 (100.0%) | 2781 (100.0%) | 4623 (100.0%) |

### Information Types
| information_type | train | test       | total        |
|:----------|------------:|------------:|-------------:|
| OTHER     | 614 (33.3%) | 848 (30.5%) | 1462 (31.6%) |
| IDENTITY  | 473 (25.7%) | 760 (27.3%) | 1233 (26.7%) |
| TEMPORAL  | 428 (23.2%) | 707 (25.4%) | 1135 (24.6%) |
| NUMERICAL | 274 (14.9%) | 388 (14.0%) | 662 (14.3%)  |
| SPATIAL   | 53 (2.9%)   | 78 (2.8%)   | 131 (2.8%)   |


### Categories

| category      | train       | test        | total        |
|:--------------|------------:|------------:|-------------:|
| SPORTS        | 485 (26.3%) | 681 (24.5%) | 1166 (25.2%) |
| POLITICS      | 408 (22.1%) | 633 (22.8%) | 1041 (22.5%) |
| MUSIC         | 235 (12.8%) | 316 (11.4%) | 551 (11.9%)  |
| MOVIES_AND_TV | 211 (11.5%) | 324 (11.7%) | 535 (11.6%)  |
| OTHER         | 200 (10.9%) | 286 (10.3%) | 486 (10.5%)  |
| BUSINESS      | 101 (5.5%)  | 153 (5.5%)  | 254 (5.5%)   |
| TECHNOLOGY    | 68 (3.7%)   | 162 (5.8%)  | 230 (5.0%)   |
| LITERATURE    | 55 (3.0%)   | 70 (2.5%)   | 125 (2.7%)   |
| SCIENCE       | 23 (1.2%)   | 63 (2.3%)   | 86 (1.9%)    |
| EDUCATION     | 31 (1.7%)   | 40 (1.4%)   | 71 (1.5%)    |
| HEALTH        | 15 (0.8%)   | 41 (1.5%)   | 56 (1.2%)    |
| FOOD          | 8 (0.4%)    | 8 (0.3%)    | 16 (0.3%)    |
| PHILOSOPHY    | 2 (0.1%)    | 4 (0.1%)    | 6 (0.1%)     |


### User Prompt Tags (User Prompt Requirements)
| tag                  | train       | test         | total        |
|:---------------------|------------:|-------------:|-------------:|
| FLAWED               | 955 (51.8%) | 1411 (50.7%) | 2366 (51.2%) |
| YOU                  | 909 (49.3%) | 1424 (51.2%) | 2333 (50.5%) |
| INSTRUCTION          | 926 (50.3%) | 1406 (50.6%) | 2332 (50.4%) |
| NON_CONFIRMATORY     | 909 (49.3%) | 1415 (50.9%) | 2324 (50.3%) |
| ADDING_CONTEXT       | 931 (50.5%) | 1389 (49.9%) | 2320 (50.2%) |
| IMMEDIATE            | 911 (49.5%) | 1392 (50.1%) | 2303 (49.8%) |
| QUESTION             | 916 (49.7%) | 1375 (49.4%) | 2291 (49.6%) |
| OBJECTIVELY          | 933 (50.7%) | 1357 (48.8%) | 2290 (49.5%) |
| PERFECT              | 887 (48.2%) | 1370 (49.3%) | 2257 (48.8%) |
| CONFIRMATORY_WRONG   | 475 (25.8%) | 678 (24.4%)  | 1153 (24.9%) |
| CONFIRMATORY_CORRECT | 458 (24.9%) | 688 (24.7%)  | 1146 (24.8%) |


### Response-Level Hallucinations by LLM

| Model                    | train        | test         | total        |
|:-------------------------|-------------:|-------------:|-------------:|
| Llama-2-7b-chat-hf       | 1007 (54.7%) | 1522 (54.7%) | 2529 (54.7%) |
| Llama-3.1-8B-Instruct    | 112 (6.1%)   | 199 (7.2%)   | 311 (6.7%)   |
| Mistral-7B-Instruct-v0.1 | 624 (33.9%)  | 891 (32.0%)  | 1515 (32.8%) |
| Mistral-7B-Instruct-v0.3 | 362 (19.7%)  | 582 (20.9%)  | 944 (20.4%)  |


## Preparing Data for SFTTrainer

<details>
<summary>Click to reveal python code</summary>

```python
from datasets import load_dataset, Dataset

# Function to prepare the prompts for fine-tuning
def formatted_dataset(dataset, max_entries=None):
    messages = []

    # Go through dataset row by row
    for ex in dataset:
        # Add chat with prompt and its golden answer to "messages"
        messages.append(
            ex["rag_prompt"] + [
            {
                "role": "assistant",
                "content": ex["responses"]["golden_answer"]
            }
        ])

        if max_entries is not None and len(messages) >= max_entries:
            break

    # Create new dataset
    return Dataset.from_dict({ "messages": messages })

# Load RAGognize
dataset = load_dataset("F4biian/RAGognize")

train_dataset = formatted_dataset(dataset["train"])
test_dataset = formatted_dataset(dataset["test"])

print(train_dataset)
print(test_dataset)
```
</details>

## Example Entry

Here is an example entry from the dataset:

<details>
<summary>Click to reveal JSON</summary>

```json
{
  "user_prompt_index": 2167,
  "user_prompt": "I understand that Christine Conley, the representative for Connecticut's District 40, resigned on January 7, 2025. I believe she left office to take a position in the private sector. Could you please verify if this was the actual reason for her resignation?",
  "answerable": true,
  "information_type": "OTHER",
  "category": "POLITICS",
  "tags": [
    "QUESTION",
    "ADDING_CONTEXT",
    "YOU",
    "PERFECT",
    "CONFIRMATORY_WRONG"
  ],
  "information_date": "2025-01-21 00:00:00",
  "documents": [
    {
      "title": "2025 Connecticut House of Representatives District 40 special election",
      "text": "representative Christine Conley won the general election in District 40 with 66.2 percent of the vote. She represented the district since 2016, and resigned from the office on January 7, 2025, to be appointed to the state Workers' Compensation Commission.\n\n\n== Candidates ==\nParty nominees for special elections"
    }
  ],
  "documents_str": "#### 2025 Connecticut House of Representatives District 40 special election\nrepresentative Christine Conley won the general election in District 40 with 66.2 percent of the vote. She represented the district since 2016, and resigned from the office on January 7, 2025, to be appointed to the state Workers' Compensation Commission.\n\n\n== Candidates ==\nParty nominees for special elections",
  "rag_prompt": [
    {
      "content": "## Request\nUser: \n<request>\nI understand that Christine Conley, the representative for Connecticut's District 40, resigned on January 7, 2025. I believe she left office to take a position in the private sector. Could you please verify if this was the actual reason for her resignation?\n</request>\n\n## passages\nUse the retrieved passages below as the sole source for your response:\n<passages>\n#### 2025 Connecticut House of Representatives District 40 special election\nrepresentative Christine Conley won the general election in District 40 with 66.2 percent of the vote. She represented the district since 2016, and resigned from the office on January 7, 2025, to be appointed to the state Workers' Compensation Commission.\n\n\n== Candidates ==\nParty nominees for special elections\n</passages>\n\n## Instructions\nGenerate a concise response that is helpful, ethical, and based entirely on the retrieved passages. Do not infer or assume details beyond what is provided. Stick to the information given and disregard anything that does not aid in answering the user’s query.",
      "role": "user"
    }
  ],
  "responses": {
    "golden_answer": "Actually, Christine Conley resigned from her position representing District 40 not to take a private sector job, but because she was appointed to the state Workers' Compensation Commission.",
    "Llama-2-7b-chat-hf": {
      "text": " Dear User,\n\nThank you for reaching out to me regarding the reason for Christine Conley's resignation from her position as representative for Connecticut's District 40. According to the passages provided, Christine Conley resigned from her office on January 7, 2025, to be appointed to the state Workers' Compensation Commission. Therefore, the actual reason for her resignation is indeed her appointment to this commission.\n\nI hope this information is helpful to you. Please let me know if you have any further questions or concerns.\n\nBest regards,\n[Your Name]",
      "hallucinations": [],
      "details": {
        "group": "train_answerable_0",
        "answerable": true,
        "full_prompt": "<s> [INST] ## Request\nUser: \n<request>\nI understand that Christine Conley, the representative for Connecticut's District 40, resigned on January 7, 2025. I believe she left office to take a position in the private sector. Could you please verify if this was the actual reason for her resignation?\n</request>\n\n## passages\nUse the retrieved passages below as the sole source for your response:\n<passages>\n#### 2025 Connecticut House of Representatives District 40 special election\nrepresentative Christine Conley won the general election in District 40 with 66.2 percent of the vote. She represented the district since 2016, and resigned from the office on January 7, 2025, to be appointed to the state Workers' Compensation Commission.\n\n\n== Candidates ==\nParty nominees for special elections\n</passages>\n\n## Instructions\nGenerate a concise response that is helpful, ethical, and based entirely on the retrieved passages. Do not infer or assume details beyond what is provided. Stick to the information given and disregard anything that does not aid in answering the user’s query. [/INST]",
        "full_chat": "<s> [INST] ## Request\nUser: \n<request>\nI understand that Christine Conley, the representative for Connecticut's District 40, resigned on January 7, 2025. I believe she left office to take a position in the private sector. Could you please verify if this was the actual reason for her resignation?\n</request>\n\n## passages\nUse the retrieved passages below as the sole source for your response:\n<passages>\n#### 2025 Connecticut House of Representatives District 40 special election\nrepresentative Christine Conley won the general election in District 40 with 66.2 percent of the vote. She represented the district since 2016, and resigned from the office on January 7, 2025, to be appointed to the state Workers' Compensation Commission.\n\n\n== Candidates ==\nParty nominees for special elections\n</passages>\n\n## Instructions\nGenerate a concise response that is helpful, ethical, and based entirely on the retrieved passages. Do not infer or assume details beyond what is provided. Stick to the information given and disregard anything that does not aid in answering the user’s query. [/INST]  Dear User,\n\nThank you for reaching out to me regarding the reason for Christine Conley's resignation from her position as representative for Connecticut's District 40. According to the passages provided, Christine Conley resigned from her office on January 7, 2025, to be appointed to the state Workers' Compensation Commission. Therefore, the actual reason for her resignation is indeed her appointment to this commission.\n\nI hope this information is helpful to you. Please let me know if you have any further questions or concerns.\n\nBest regards,\n[Your Name]</s>",
        "output": " Dear User,\n\nThank you for reaching out to me regarding the reason for Christine Conley's resignation from her position as representative for Connecticut's District 40. According to the passages provided, Christine Conley resigned from her office on January 7, 2025, to be appointed to the state Workers' Compensation Commission. Therefore, the actual reason for her resignation is indeed her appointment to this commission.\n\nI hope this information is helpful to you. Please let me know if you have any further questions or concerns.\n\nBest regards,\n[Your Name]",
        "token_starts": [
          0, 0, 2, 5, 10, 11, 12, 13, 18, 22, 26, 35, 39, 42, 45, 55, 59, 66, 70, 77, 80, 84, 87, 88, 89, 93, 101, 106, 110, 119, 122, 137, 141, 153, 154, 155, 164, 165, 166, 167, 168, 178, 181, 185, 190, 194, 203, 204, 211, 214, 218, 221, 225, 230, 235, 239, 246, 249, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 269, 272, 282, 285, 289, 295, 300, 303, 304, 309, 312, 317, 328, 329, 339, 340, 344, 351, 358, 362, 366, 370, 378, 381, 388, 392, 404, 407, 412, 423, 424, 425, 426, 427, 432, 437, 449, 452, 460, 463, 467, 468, 475, 479, 482, 487, 490, 494, 499, 503, 511, 521, 524, 533, 534, 535, 536, 540, 548, 549, 550, 551, 555, 560, 561
        ],
        "annotations": {
          "original_output": {
            "hallucinations_analysis": "There are no hallucinations in the response. The LLM correctly identified the reason for Christine Conley's resignation as being appointed to the state Workers' Compensation Commission, which is directly stated in the provided passages.",
            "hallucination_as_a_whole_analysis": "The entire response is not hallucinated. The LLM accurately extracted the reason for resignation from the provided passages.",
            "completely_hallucinated": false,
            "partially_hallucinated": false,
            "hallucinations": [],
            "cluelessness_analysis": "The LLM did not state cluelessness.",
            "cluelessness": false,
            "addressed_user_request_analysis": "The LLM directly addressed the user's query about the reason for Christine Conley's resignation, stating the reason found in the passages.",
            "addressed_user_request": true
          },
          "result": {
            "all_valid": true,
            "addressed_user_prompt": true,
            "cluelessness": false,
            "completely_hallucinated": false,
            "hallucinations": []
          }
        }
      }
    },
    "Llama-3.1-8B-Instruct": {
      "text": "Based on the provided information, Christine Conley resigned from her position as representative for Connecticut's District 40 on January 7, 2025, to be appointed to the state Workers' Compensation Commission. The reason for her resignation is not to take a position in the private sector, but rather to take a role in the state's Workers' Compensation Commission.",
      "hallucinations": [],
      "details": {
        "group": "train_answerable_0",
        "answerable": true,
        "full_prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n## Request\nUser: \n<request>\nI understand that Christine Conley, the representative for Connecticut's District 40, resigned on January 7, 2025. I believe she left office to take a position in the private sector. Could you please verify if this was the actual reason for her resignation?\n</request>\n\n## passages\nUse the retrieved passages below as the sole source for your response:\n<passages>\n#### 2025 Connecticut House of Representatives District 40 special election\nrepresentative Christine Conley won the general election in District 40 with 66.2 percent of the vote. She represented the district since 2016, and resigned from the office on January 7, 2025, to be appointed to the state Workers' Compensation Commission.\n\n\n== Candidates ==\nParty nominees for special elections\n</passages>\n\n## Instructions\nGenerate a concise response that is helpful, ethical, and based entirely on the retrieved passages. Do not infer or assume details beyond what is provided. Stick to the information given and disregard anything that does not aid in answering the user’s query.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "full_chat": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n## Request\nUser: \n<request>\nI understand that Christine Conley, the representative for Connecticut's District 40, resigned on January 7, 2025. I believe she left office to take a position in the private sector. Could you please verify if this was the actual reason for her resignation?\n</request>\n\n## passages\nUse the retrieved passages below as the sole source for your response:\n<passages>\n#### 2025 Connecticut House of Representatives District 40 special election\nrepresentative Christine Conley won the general election in District 40 with 66.2 percent of the vote. She represented the district since 2016, and resigned from the office on January 7, 2025, to be appointed to the state Workers' Compensation Commission.\n\n\n== Candidates ==\nParty nominees for special elections\n</passages>\n\n## Instructions\nGenerate a concise response that is helpful, ethical, and based entirely on the retrieved passages. Do not infer or assume details beyond what is provided. Stick to the information given and disregard anything that does not aid in answering the user’s query.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nBased on the provided information, Christine Conley resigned from her position as representative for Connecticut's District 40 on January 7, 2025, to be appointed to the state Workers' Compensation Commission. The reason for her resignation is not to take a position in the private sector, but rather to take a role in the state's Workers' Compensation Commission.<|eot_id|>",
        "output": "Based on the provided information, Christine Conley resigned from her position as representative for Connecticut's District 40 on January 7, 2025, to be appointed to the state Workers' Compensation Commission. The reason for her resignation is not to take a position in the private sector, but rather to take a role in the state's Workers' Compensation Commission.",
        "token_starts": [
          0, 5, 8, 12, 21, 33, 34, 44, 48, 51, 60, 65, 69, 78, 81, 96, 100, 112, 114, 123, 124, 126, 129, 137, 138, 139, 140, 141, 144, 145, 146, 149, 152, 162, 165, 169, 175, 183, 184, 197, 208, 209, 213, 220, 224, 228, 240, 243, 247, 250, 255, 257, 266, 269, 273, 281, 288, 289, 293, 300, 303, 308, 310, 315, 318, 322, 328, 330, 338, 339, 352, 363, 364
        ],
        "annotations": {
          "original_output": {
            "hallucinations_analysis": "There are no hallucinations in the response. The LLM accurately extracts the reason for resignation from the provided text and uses it to answer the user's question and correct their assumption.",
            "hallucination_as_a_whole_analysis": "The entire response is not hallucinated. The LLM accurately extracts the reason for resignation from the provided text and uses it to answer the user's question and correct their assumption.",
            "completely_hallucinated": false,
            "partially_hallucinated": false,
            "hallucinations": [],
            "cluelessness_analysis": "The LLM does not state cluelessness.",
            "cluelessness": false,
            "addressed_user_request_analysis": "The LLM directly addresses the user's query by stating the reason for Christine Conley's resignation as provided in the text and clarifying that it was not for a private sector position as the user hypothesized.",
            "addressed_user_request": true
          },
          "result": {
            "all_valid": true,
            "addressed_user_prompt": true,
            "cluelessness": false,
            "completely_hallucinated": false,
            "hallucinations": []
          }
        }
      }
    },
    "Mistral-7B-Instruct-v0.1": {
      "text": "According to the retrieved passages, Christine Conley, the representative for Connecticut's District 40, resigned from her office on January 7, 2025, to be appointed to the state Workers' Compensation Commission. The reason for her resignation was not specified in the passages.",
      "hallucinations": [
        {
          "valid": true,
          "text": "The reason for her resignation was not specified in the passages.",
          "start": 213,
          "end": 278
        }
      ],
      "details": {
        "group": "train_answerable_0",
        "answerable": true,
        "full_prompt": "<s> [INST] ## Request\nUser: \n<request>\nI understand that Christine Conley, the representative for Connecticut's District 40, resigned on January 7, 2025. I believe she left office to take a position in the private sector. Could you please verify if this was the actual reason for her resignation?\n</request>\n\n## passages\nUse the retrieved passages below as the sole source for your response:\n<passages>\n#### 2025 Connecticut House of Representatives District 40 special election\nrepresentative Christine Conley won the general election in District 40 with 66.2 percent of the vote. She represented the district since 2016, and resigned from the office on January 7, 2025, to be appointed to the state Workers' Compensation Commission.\n\n\n== Candidates ==\nParty nominees for special elections\n</passages>\n\n## Instructions\nGenerate a concise response that is helpful, ethical, and based entirely on the retrieved passages. Do not infer or assume details beyond what is provided. Stick to the information given and disregard anything that does not aid in answering the user’s query. [/INST]",
        "full_chat": "<s> [INST] ## Request\nUser: \n<request>\nI understand that Christine Conley, the representative for Connecticut's District 40, resigned on January 7, 2025. I believe she left office to take a position in the private sector. Could you please verify if this was the actual reason for her resignation?\n</request>\n\n## passages\nUse the retrieved passages below as the sole source for your response:\n<passages>\n#### 2025 Connecticut House of Representatives District 40 special election\nrepresentative Christine Conley won the general election in District 40 with 66.2 percent of the vote. She represented the district since 2016, and resigned from the office on January 7, 2025, to be appointed to the state Workers' Compensation Commission.\n\n\n== Candidates ==\nParty nominees for special elections\n</passages>\n\n## Instructions\nGenerate a concise response that is helpful, ethical, and based entirely on the retrieved passages. Do not infer or assume details beyond what is provided. Stick to the information given and disregard anything that does not aid in answering the user’s query. [/INST] According to the retrieved passages, Christine Conley, the representative for Connecticut's District 40, resigned from her office on January 7, 2025, to be appointed to the state Workers' Compensation Commission. The reason for her resignation was not specified in the passages.</s>",
        "output": "According to the retrieved passages, Christine Conley, the representative for Connecticut's District 40, resigned from her office on January 7, 2025, to be appointed to the state Workers' Compensation Commission. The reason for her resignation was not specified in the passages.",
        "token_starts": [
          0, 9, 12, 16, 24, 26, 31, 35, 36, 43, 46, 50, 53, 54, 58, 73, 77, 89, 90, 91, 100, 101, 102, 103, 104, 113, 118, 122, 129, 132, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 152, 155, 165, 168, 172, 178, 183, 186, 187, 192, 195, 200, 211, 212, 216, 223, 227, 231, 238, 243, 247, 251, 261, 264, 268, 273, 277, 278
        ],
        "annotations": {
          "original_output": {
            "hallucinations_analysis": "The LLM states that the reason for Christine Conley's resignation was not specified in the passages. However, the provided passage explicitly states that she resigned \"to be appointed to the state Workers' Compensation Commission.\" This directly contradicts the LLM's claim that the reason was not specified. Therefore, the statement \"The reason for her resignation was not specified in the passages\" is a hallucination.",
            "hallucination_as_a_whole_analysis": "The entire response is not a hallucination. The first sentence accurately reflects information found in the provided passages.",
            "completely_hallucinated": false,
            "partially_hallucinated": true,
            "hallucinations": [
              {
                "substring_with_context": "Commission. The reason for her resignation was not specified in the passages.",
                "substring_analysis": "The substring \"The reason for her resignation was not specified in the passages.\" accurately captures the hallucinated information. The LLM claims the reason was not specified, but the passage explicitly states the reason.",
                "final_substring": "The reason for her resignation was not specified in the passages."
              }
            ],
            "cluelessness_analysis": "The LLM provided information based on the passages.",
            "cluelessness": false,
            "addressed_user_request_analysis": "The LLM attempted to address the user's query about the reason for Christine Conley's resignation.",
            "addressed_user_request": true
          },
          "result": {
            "all_valid": true,
            "addressed_user_prompt": true,
            "cluelessness": false,
            "completely_hallucinated": false,
            "hallucinations": [
              {
                "valid": true,
                "text": "The reason for her resignation was not specified in the passages.",
                "start": 213,
                "end": 278
              }
            ]
          }
        }
      }
    },
    "Mistral-7B-Instruct-v0.3": {
      "text": "Based on the provided passages, Christine Conley resigned from her position as the representative for Connecticut's District 40 on January 7, 2025, to be appointed to the state Workers' Compensation Commission. There is no mention of her leaving office to take a position in the private sector.",
      "hallucinations": [],
      "details": {
        "group": "train_answerable_0",
        "answerable": true,
        "full_prompt": "<s>[INST] ## Request\nUser: \n<request>\nI understand that Christine Conley, the representative for Connecticut's District 40, resigned on January 7, 2025. I believe she left office to take a position in the private sector. Could you please verify if this was the actual reason for her resignation?\n</request>\n\n## passages\nUse the retrieved passages below as the sole source for your response:\n<passages>\n#### 2025 Connecticut House of Representatives District 40 special election\nrepresentative Christine Conley won the general election in District 40 with 66.2 percent of the vote. She represented the district since 2016, and resigned from the office on January 7, 2025, to be appointed to the state Workers' Compensation Commission.\n\n\n== Candidates ==\nParty nominees for special elections\n</passages>\n\n## Instructions\nGenerate a concise response that is helpful, ethical, and based entirely on the retrieved passages. Do not infer or assume details beyond what is provided. Stick to the information given and disregard anything that does not aid in answering the user’s query.[/INST]",
        "full_chat": "<s>[INST] ## Request\nUser: \n<request>\nI understand that Christine Conley, the representative for Connecticut's District 40, resigned on January 7, 2025. I believe she left office to take a position in the private sector. Could you please verify if this was the actual reason for her resignation?\n</request>\n\n## passages\nUse the retrieved passages below as the sole source for your response:\n<passages>\n#### 2025 Connecticut House of Representatives District 40 special election\nrepresentative Christine Conley won the general election in District 40 with 66.2 percent of the vote. She represented the district since 2016, and resigned from the office on January 7, 2025, to be appointed to the state Workers' Compensation Commission.\n\n\n== Candidates ==\nParty nominees for special elections\n</passages>\n\n## Instructions\nGenerate a concise response that is helpful, ethical, and based entirely on the retrieved passages. Do not infer or assume details beyond what is provided. Stick to the information given and disregard anything that does not aid in answering the user’s query.[/INST] Based on the provided passages, Christine Conley resigned from her position as the representative for Connecticut's District 40 on January 7, 2025, to be appointed to the state Workers' Compensation Commission. There is no mention of her leaving office to take a position in the private sector.</s>",
        "output": "Based on the provided passages, Christine Conley resigned from her position as the representative for Connecticut's District 40 on January 7, 2025, to be appointed to the state Workers' Compensation Commission. There is no mention of her leaving office to take a position in the private sector.",
        "token_starts": [
          0, 5, 8, 12, 21, 26, 30, 31, 38, 41, 45, 48, 57, 62, 66, 75, 78, 82, 97, 101, 113, 114, 115, 124, 125, 126, 127, 130, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 150, 153, 163, 166, 170, 176, 181, 184, 185, 190, 193, 198, 209, 210, 216, 219, 222, 230, 233, 237, 245, 252, 255, 260, 262, 271, 274, 278, 286, 293, 294
        ],
        "annotations": {
          "original_output": {
            "hallucinations_analysis": "There are no hallucinations in the LLM's response. All information provided is directly supported by the retrieved passages.",
            "hallucination_as_a_whole_analysis": "The entire response is not hallucinated. The LLM accurately reports the reason for resignation as stated in the provided passages and correctly notes that the passages do not mention the private sector reason.",
            "completely_hallucinated": false,
            "partially_hallucinated": false,
            "hallucinations": [],
            "cluelessness_analysis": "The LLM does not state cluelessness or lack of information.",
            "cluelessness": false,
            "addressed_user_request_analysis": "The LLM directly addresses the user's request by stating the reason for Christine Conley's resignation according to the provided passages and confirming that the passages do not mention the private sector reason suggested by the user.",
            "addressed_user_request": true
          },
          "result": {
            "all_valid": true,
            "addressed_user_prompt": true,
            "cluelessness": false,
            "completely_hallucinated": false,
            "hallucinations": []
          }
        }
      }
    }
  },
  "details": {
    "amount_of_chunks": 1,
    "answerable_chunks": [
      {
        "passage_index": null,
        "suitable_article_index": null,
        "suitable_article_title": "2025 Connecticut House of Representatives District 40 special election",
        "text": "representative Christine Conley won the general election in District 40 with 66.2 percent of the vote. She represented the district since 2016, and resigned from the office on January 7, 2025, to be appointed to the state Workers' Compensation Commission.\n\n\n== Candidates ==\nParty nominees for special elections"
      }
    ],
    "info_answerable": "#### 2025 Connecticut House of Representatives District 40 special election\nrepresentative Christine Conley won the general election in District 40 with 66.2 percent of the vote. She represented the district since 2016, and resigned from the office on January 7, 2025, to be appointed to the state Workers' Compensation Commission.\n\n\n== Candidates ==\nParty nominees for special elections",
    "info_unanswerable": "#### Ursula Jones Dickson\nley, the San Francisco Chronicle, and all 14 police unions in Alameda County. Price left office on December 5, 2024, and her chief assistant Royl Roberts took over as acting district attorney.\nThe Alameda County Board of Supervisors selected seven finalists from a pool of 15 applicants to consider for the vacancy. On January 28, 2025, after more than a month of deliberation, the Board of Supervisors appointed Jones Dickson, assuming office on February 18, 2025. She stated her first action while in office would be to rescind a policy, established by her predecessor, that required prosecutors to seek approval from the office’s leaders before filing sentencing enhancements.\n\n\n== Personal life ==\nThough her roles are officially nonparti",
    "source_chunk_index": 0,
    "template": [
      {
        "content": "## Request\nUser: \n<request>\n{{question}}\n</request>\n\n## passages\nUse the retrieved passages below as the sole source for your response:\n<passages>\n{{info}}\n</passages>\n\n## Instructions\nGenerate a concise response that is helpful, ethical, and based entirely on the retrieved passages. Do not infer or assume details beyond what is provided. Stick to the information given and disregard anything that does not aid in answering the user’s query.",
        "role": "user"
      }
    ],
    "template_details": {
      "sample_log": {
        "answer_suffix": false,
        "info_label": {
          "plural": "passages",
          "singular": "passage"
        },
        "info_prov": "Use the retrieved {{info_label_plural}} below as the sole source for your response:\n{{info}}",
        "instr_before_info": false,
        "instructions": "Generate a concise response that is helpful, ethical, and based entirely on the retrieved {{info_label_plural}}. Do not infer or assume details beyond what is provided. Stick to the information given and disregard anything that does not aid in answering the user’s query.",
        "md_capitalize": false,
        "md_hashtags": "##",
        "md_info_prov_title": "{{info_label_plural}}",
        "md_instr_title": "Instructions",
        "prompt_format": "MARKDOWN",
        "question_phrase": "User: {{question}}",
        "section_emb": "TAGS",
        "system_message": null,
        "unansw_hint": ""
      },
      "settings": {
        "answer_suffix": false,
        "has_sys_msg": false,
        "info_in_sys_msg": false,
        "instructions_in_sys_msg": false,
        "question_after_info": false,
        "unanswerability_hint": false,
        "unanswerability_hint_in_sys_msg": false
      }
    },
    "unanswerable_chunks": [
      {
        "passage_index": 1,
        "suitable_article_index": 1124,
        "suitable_article_title": "Ursula Jones Dickson",
        "text": "ley, the San Francisco Chronicle, and all 14 police unions in Alameda County. Price left office on December 5, 2024, and her chief assistant Royl Roberts took over as acting district attorney.\nThe Alameda County Board of Supervisors selected seven finalists from a pool of 15 applicants to consider for the vacancy. On January 28, 2025, after more than a month of deliberation, the Board of Supervisors appointed Jones Dickson, assuming office on February 18, 2025. She stated her first action while in office would be to rescind a policy, established by her predecessor, that required prosecutors to seek approval from the office’s leaders before filing sentencing enhancements.\n\n\n== Personal life ==\nThough her roles are officially nonparti"
      }
    ],
    "user_prompt": {
      "article_title": "2025 Connecticut House of Representatives District 40 special election",
      "category": "POLITICS",
      "cluelessness_answer": "I am sorry, but I don't have the specific details regarding the reason for Christine Conley's resignation from Connecticut's District 40.",
      "details": {
        "answer_quote_from_passage": "to be appointed to the state Workers' Compensation Commission",
        "passage_index": 0,
        "suitable_article": {
          "backlinks": 21,
          "content": "A special election was held in the U.S. state of Connecticut to elect a new member for District 40 in the Connecticut House of Representatives, representing parts of the towns of Groton and Ledyard in New London County. The election filled a vacancy caused by the resignation of Democratic representative Christine Conley on January 7, 2025. Dan Gaiewski, a Democratic member of the Groton Town Council, was elected representative in the special election. The election took place concurrently with a special election in Senate District 21.\n\n\n== Background ==\nIn 2024, incumbent Democratic representative Christine Conley won the general election in District 40 with 66.2 percent of the vote. She represented the district since 2016, and resigned from the office on January 7, 2025, to be appointed to the state Workers' Compensation Commission.\n\n\n== Candidates ==\nParty nominees for special elections are chosen through party conventions. Political parties were required to submit names for the ballot by February 15, 2025.\n\nDan Gaiewski, member of the Groton Town Council (Democratic, Working Families)\nRobert Boris, chairman of the Mystic Chamber Foundation (Republican)\n\n\n== Results ==\n\n\n== See also ==\n2025 United States state legislative elections\n\n\n== References ==\n\n\n== External links ==\nElections and Voter Information (Connecticut Secretary of State)\nConnecticut's 40th House of Representatives district (Ballotpedia)\n\n\n=== Official campaign websites ===\nDan Gaiewski (D)\nRobert Boris (R)",
          "created_en": "2025-02-22 23:41:00",
          "earliest_creation_date": null,
          "passage_data": [
            {
              "contains_article": false,
              "earliest_access_date": "2025-02-22 00:00:00",
              "earliest_archive_date": null,
              "earliest_date": "2025-01-21 00:00:00",
              "end": 844,
              "references": [
                {
                  "access_date": "2025-02-22 00:00:00",
                  "archive_date": null,
                  "date": "2025-01-21 00:00:00",
                  "key_count": 8,
                  "ref_label": "cite news"
                }
              ],
              "start": 692
            }
          ],
          "removed_duplicates": false,
          "retrieval_date_utc": "2025-02-28 14:28:38.559233",
          "revision_id": 1277757336,
          "title": "2025 Connecticut House of Representatives District 40 special election",
          "url": "https://en.wikipedia.org/wiki/2025_Connecticut_House_of_Representatives_District_40_special_election"
        },
        "suitable_article_index": 2219,
        "user_prompt_requirements": [
          "QUESTION",
          "ADDING_CONTEXT",
          "YOU",
          "PERFECT",
          "CONFIRMATORY_WRONG"
        ]
      },
      "golden_answer": "Actually, Christine Conley resigned from her position representing District 40 not to take a private sector job, but because she was appointed to the state Workers' Compensation Commission.",
      "information_type": "OTHER",
      "passage_containing_answer": "She represented the district since 2016, and resigned from the office on January 7, 2025, to be appointed to the state Workers' Compensation Commission.",
      "user_prompt": "I understand that Christine Conley, the representative for Connecticut's District 40, resigned on January 7, 2025. I believe she left office to take a position in the private sector. Could you please verify if this was the actual reason for her resignation?",
      "user_prompt_index": 2167
    }
  }
}
```

</details>

## Potential Limitations and Biases
**Risk of Introducing Bias And Misinformation:** The data retrieved from Wikipedia has never been validated and checked for truthfulness. Since the dataset contains a good amount of political information, there is a possibility that retrieval sources may embed political bias. Users should take this and other risks into account when fine-tuning LLMs on this dataset.

## Citation

If you use this dataset in your work, please cite it as follows:

```bibtex
@misc{ridder2026ragognizerhallucinationawarefinetuningdetection,
      title={RAGognizer: Hallucination-Aware Fine-Tuning via Detection Head Integration}, 
      author={Fabian Ridder and Laurin Lessel and Malte Schilling},
      year={2026},
      eprint={2604.15945},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.15945}, 
}
```

## Licensing Information

This dataset was derived from Wikipedia content, which is licensed under the
[Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

© Wikimedia Foundation and Wikipedia contributors.

The dataset itself is distributed under the same
[CC BY-SA 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/).

Modifications: The dataset has been processed and reformatted from the original Wikipedia text.