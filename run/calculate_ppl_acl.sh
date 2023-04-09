python3 calculate_ppl_acl.py \
--model_dir out_wikipedia_en \
--device cuda:0 \
--minimum_context_length 512 \
--computing_method long_history \

python3 calculate_ppl_acl.py \
--model_dir out_wikipedia_en \
--device cuda:0 \
--computing_method naive \

python3 calculate_ppl_acl.py \
--model_dir out_openwebtext \
--device cuda:0 \
--minimum_context_length 512 \
--computing_method long_history \

python3 calculate_ppl_acl.py \
--model_dir out_wikipedia_en \
--device cuda:0 \
--minimum_context_length 256 \
--computing_method long_history \

python3 calculate_ppl_acl.py \
--model_dir out_wikipedia_en \
--device cuda:0 \
--minimum_context_length 768 \
--computing_method long_history \
