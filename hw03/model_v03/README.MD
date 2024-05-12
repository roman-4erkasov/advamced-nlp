# Первое улучшение
- Используем леммы

fatuus@ubuntu22:~/advanced-nlp/hw03$ ve/bin/python ./model_v01/word_alignment.py training/en-cs.en.all.tokens.10K training/en-cs.cs.all.lemmas.10K 20 mdl00_lemm
corpus log likelihood: -1101394.999
corpus log likelihood: -895863.243
corpus log likelihood: -785560.353
corpus log likelihood: -743353.593
corpus log likelihood: -726944.618
corpus log likelihood: -719690.519
corpus log likelihood: -715946.795
corpus log likelihood: -713775.479
corpus log likelihood: -712412.707
corpus log likelihood: -711509.262
corpus log likelihood: -710876.240
corpus log likelihood: -710413.269
corpus log likelihood: -710063.579
corpus log likelihood: -709794.194
corpus log likelihood: -709583.137
corpus log likelihood: -709416.282
corpus log likelihood: -709281.957
corpus log likelihood: -709170.528
corpus log likelihood: -709075.956
fatuus@ubuntu22:~/advanced-nlp/hw03$ ./ve/bin/python eval.py ./test/en-cs.en.dev.tokens ./test/en-cs.cs.dev.lemmas ./test/en-cs.wa.dev ./mdl00_lemm.dev.wa
recall 0.557; precision 0.590; aer 0.427




fatuus@ubuntu22:~/advanced-nlp/hw03$ ve/bin/python ./model_v01/word_alignment.py training/en-cs.en.all.tokens.10K training/en-cs.cs.all.lemmas.10K 40 mdl01
corpus log likelihood: -1101394.999
corpus log likelihood: -895863.243
corpus log likelihood: -785560.353
corpus log likelihood: -743353.593
corpus log likelihood: -726944.618
corpus log likelihood: -719690.519
corpus log likelihood: -715946.795
corpus log likelihood: -713775.479
corpus log likelihood: -712412.707
corpus log likelihood: -711509.262
corpus log likelihood: -710876.240
corpus log likelihood: -710413.269
corpus log likelihood: -710063.579
corpus log likelihood: -709794.194
corpus log likelihood: -709583.137
corpus log likelihood: -709416.282
corpus log likelihood: -709281.957
corpus log likelihood: -709170.528
corpus log likelihood: -709075.956
corpus log likelihood: -708995.117
corpus log likelihood: -708925.558
corpus log likelihood: -708865.531
corpus log likelihood: -708813.578
corpus log likelihood: -708768.128
corpus log likelihood: -708727.757
corpus log likelihood: -708692.011
corpus log likelihood: -708660.756
corpus log likelihood: -708633.522
corpus log likelihood: -708609.411
corpus log likelihood: -708587.762
corpus log likelihood: -708568.357
corpus log likelihood: -708551.086
corpus log likelihood: -708535.597
corpus log likelihood: -708521.670
corpus log likelihood: -708509.046
corpus log likelihood: -708497.444
corpus log likelihood: -708486.745
corpus log likelihood: -708476.902
corpus log likelihood: -708467.852