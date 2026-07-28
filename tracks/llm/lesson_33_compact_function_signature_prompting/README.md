# Lesson 33: Compact Function-Signature Prompting

This lesson teaches function-signature prompting with a tiny synthetic causal LM
task. Each example provides a function name, argument slots, argument types, and
an output marker. Supervision begins only at the `call` marker so the model
learns to continue with a signature-compatible function invocation.
