# NLP_web_API


Required Input parameters for all functions 


sku_id (optional leave blank if not needed)


customer 


comment_time_start


comment_time_end


table name





predict method (POST requests): server_IP:50001/nlp_return/predict

returns predictions on comments.


word_freq method (POST requests): server_IP:50001/nlp_return/word_freq

returns the 20 most common words in negative and positive comments seperately.
