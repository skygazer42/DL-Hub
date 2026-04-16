# Lesson 34: Toy XML Constrained Prompting

This lesson teaches structured continuation under a simple XML-style output constraint. Each
synthetic sequence begins with a short prompt and schema header, then switches at an explicit
`xml_token_id` marker into the supervised XML payload region.

The dataset masks all prompt-side tokens to `ignore_index` so training starts exactly at the XML
boundary. A small causal transformer predicts the structured tail and writes the standard lesson
artifacts for smoke testing.

Run:

```bash
python -m tracks.llm.lesson_34_toy_xml_constrained_prompting.train --epochs 1 --device cpu
```

Outputs are written under
`outputs/llm/lesson_34_toy_xml_constrained_prompting/<run_name>/`.
