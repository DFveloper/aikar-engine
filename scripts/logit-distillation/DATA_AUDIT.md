# Korean conversation audit

Date: 2026-11-04. Source: /mnt/openwebui/AIKAR/train_ko.jsonl. JSONL line numbers below are one-based; message positions are zero-based.

The file parses as 317 JSON objects with messages arrays. Of these, 209 contain one user/assistant pair, 50 contain two pairs and 58 contain three pairs. Alternating role labels alone do not detect the content errors. Inspection finds that later user fields sometimes contain translated assistant answers instead of the follow-up question, and some subsequent assistant responses contain extraction errors or are empty.

Examples:

- Line 1, message 2 is a long assistant-style explanation beginning with an acknowledgement of a fair point and a detailed Fukushima answer. Message 3 repeats the same answer with slightly different introductory wording. Message 4 is another assistant-style explanation labelled as user.
- Line 7, message 2 is a TOCSY explanatory answer labelled as user; the following assistant content is empty.
- Line 13, message 2 is a long Kafka/Pulsar/NATS comparison labelled as user; the following assistant says the requested JSON pointer path was not included.

There are 108 multi-turn conversations and 166 later user messages. A length/prefix heuristic marks many suspicious examples, but it is not a reliable automatic corruption detector and is not used to selectively repair conversations. The user's chosen fallback is applied uniformly: keep only the first user/assistant pair of every Korean row, retaining an initial system message if present and all fields of the first assistant including reasoning. Each first user and assistant content is required to be nonempty. Source files are unchanged.

The prepared mixed-train-ko-first-1to1.jsonl contains 317 Korean first-turn records plus 317 English records sampled without replacement from 120342 source records, using seed 42. It has 634 rows and 4950283 bytes on this machine. English conversations are preserved as sampled. Every Korean first-turn pair and reasoning was checked against its original row. A synthetic test also verifies system-message preservation, later-turn removal, deterministic sampling and rejection of a changed seed for an existing output.

Use a new teacher cache with this dataset. The mixed file name and manifest policy flag differ from the earlier full-multiturn mixture, and production manifest checks reject reuse of the earlier inputs. Earlier CPU speed measurements still describe their historical sample, but corpus-size estimates must use the new plan: 4493 training windows and 32 validation windows at context 512, with a 6609698365-byte cache.
