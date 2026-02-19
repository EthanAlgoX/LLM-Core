# Terminology Glossary

Use this file as the single source of truth for preferred terms.

## How To Use

- Add new domain terms before large doc updates.
- Use the `Canonical Term` in all docs.
- Put discouraged variants in `Avoid / Alias`.

## Glossary

| Canonical Term | Avoid / Alias | Type | Definition | Notes |
| --- | --- | --- | --- | --- |
| LLM | Large Language Model (after first definition) | Acronym | Autoregressive language model family for text understanding and generation. | Define once per document on first mention. |
| VLM | Vision-Language Model | Acronym | Multimodal model that combines visual and textual inputs. | Keep hyphenation consistent: `Vision-Language`. |
| RLHF | RL from Human Feedback | Acronym | Alignment pipeline using preference data and reinforcement learning. | If expanded, use `Reinforcement Learning from Human Feedback`. |
| DPO | Direct Preference Optimization | Acronym | Offline preference optimization without explicit reward model training loop. | Do not mix with PPO unless comparing methods. |
| GRPO | Group Relative Policy Optimization | Acronym | Relative advantage optimization based on grouped candidate responses. | Keep expansion exact. |
| KV Cache | Key-Value cache | Term | Attention cache used to speed up autoregressive decoding. | Use `KV Cache` consistently in headings and text. |
| PagedAttention | paged attention | Term | Memory management strategy for efficient KV cache serving. | Keep product-style capitalization: `PagedAttention`. |

## Change Log Template

When adding or changing terms, append an entry:

| Date (YYYY-MM-DD) | Change | Owner |
| --- | --- | --- |
| 2026-02-19 | Initial glossary scaffold | docs-maintainer |

