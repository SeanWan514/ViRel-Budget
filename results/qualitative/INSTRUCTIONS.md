# Blinded qualitative annotation

Review each original/intervened pair without opening `private_blind_mapping.csv`.

1. Judge whether the intervention is eligible for the question: it should meaningfully remove,
   alter, or replace visual evidence relevant to answering the question.
2. Answer the question independently for the original and intervened image.
3. Mark visible artifacts that could cause a response change for reasons unrelated to the intended
   intervention.
4. Use `uncertain` rather than forcing an answer.
5. Annotators A and B work independently. Resolve disagreements only after both forms are frozen.

The method, model, token budget, controller outcome, and success/failure category are blinded.
