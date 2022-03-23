# Debug logs 

setting up generics-kb ours: (adding a new configuration)
- install `datasets` from source to make it easily editable: `pip install --user -e .` 
- kept on saying that my configuration was not part of the configurations despite being there. 
    - needed to run `datasets-cli test datasets/generics_kb --save_infos --all_configs` to enroll my new configuration before I could load it easily with `load_dataset('generics_kb', 'generics_kb_ours')`
    - if problem persists in that my configuration is not found, cache name once in datasets dir: `load_dataset('datasets/generics_kb', 'generics_kb_ours')` --> force load from local dir instead of checking remote hub first. 
    - *note wished local dir was prioritized first
- need some commenting out to do (manual download check, waterloo configuration problematic, etc.)