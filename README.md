# llm_project

For this project LLM models are retrained using a series of preprocessed tweets to generate hashtags. 

## Dependencies
* Python Libraries
  * transformers
  * datasets
  * pandas
  * matplotlib
  * json
  * re
  * csv
 
## To run the model
* tweet_processor.py
  * have `tweets.json` in same directory as script
  * `python3 tweet_processor.py`
  * creates `tweets.csv` in that directory with output
* trainer.py
  * have `tweets.csv` in the same directory
  * `python3 trainer.py`
  * creates `./finetuned_model` as a new directory with the new finetuned model
* model.py
  * have `./finetuned_model` in the same directory
  * `python3 model.py`
  * outputs prediction directly to the console
