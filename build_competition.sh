rm competition.zip
cd competition/reference_data && zip -r reference_data.zip . && mv reference_data.zip ../ && cd ../../
cd competition/scoring_program && zip -r scoring_program.zip . && mv scoring_program.zip ../ && cd ../../
cd competition && zip -r competition.zip . && mv competition.zip ../ && cd ../
rm competition/reference_data.zip
rm competition/scoring_program.zip
