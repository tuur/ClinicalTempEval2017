# Author: Artuur Leeuwenberg
# Date: 21-02-2018
# Email: tuur.leeuwenberg@cs.kuleuven.be

# PLEASE CITE IN CASE OF USAGE:
# Artuur Leeuwenberg, and Marie-Francine Moens. KULeuven-LIIR at SemEval 2017 task 12: Cross-domain temporal information extraction from clinical records. In Proceedings of SemEval. Vancouver, Canada: ACL. (2017)
# Artuur Leeuwenberg, and Marie-Francine Moens. Structured learning for temporal relation extraction from clinical records. In Proceedings of EACL. Valencia, Spain: ACL. (2017)

# ---------------------------------------------------------------------------------------------------------
# Running the full temporal information extraction pipeline: EVENT and TIMEX3 extraction + TLINK extraction
# ---------------------------------------------------------------------------------------------------------

# YOUR SETTINGS
# ------------->

# Data Directories
TRAIN_XML_AND_TXT='./mini_silver_data/train/'
TEST_TXT='./mini_silver_data/test/'
TEST_XML_AVAILABLE=0
OUT_DIR='./output'

# Using cTakes features (optional)
TRAIN_CTAKES_XML='None'
TEST_CTAKES_XML='None'

# Hyperparameters
lowercase=1
digit_conflation=1
unk_token=1
pos=0
tlinks=CONTAINS

# PREPROCESSING & MODEL TRAINING
# ------------------------------>

# Working and output directories
MODEL_DIR=$OUT_DIR'/models/'
PRED_E=$OUT_DIR'/predictions/entities'
PRED_R=$OUT_DIR'/predictions/entities+relations'

# Training entity extraction
echo '>>>> TRAINING ENTITY SPAN+ATTRIBUTE EXTRACTION'
python main_entities.py $TRAIN_XML_AND_TXT -train_models $MODEL_DIR -unk_token $unk_token -conflate_digits $digit_conflation -pos $pos -lowercase $lowercase

# Training relation extraction
echo '>>>> TRAINING RELATION EXTRACTION'
python main_rels.py $TRAIN_XML_AND_TXT -sp 1 -p 0 -it 32 -constraints MUL -train_model $MODEL_DIR -averaging 1 -local_initialization 0 -negative_subsampling 'loss_augmented'  -lr 1 -constraint_setting CC -decreasing_lr 0 -tlinks $tlinks -lowercase $lowercase -unk_token $unk_token -conflate_digits $digit_conflation -pos $pos -ctakes_out_dir $TRAIN_CTAKES_XML


# PREDICTION OF RAW TXT
# ---------------------

# Prediction of entity spans
echo '>>>> PREDICTING ENTITIES'
python main_entities.py $TEST_TXT -test_models $MODEL_DIR -test_xml $TEST_XML_AVAILABLE -to_anafora $PRED_E -unk_token $unk_token -conflate_digits $digit_conflation -lowercase $lowercase -pos $pos


# writing text files to entity output folder as well (needed for predicting relations afterwards)
echo '>>>> COPYING TXT FILES'
for i in $TEST_TXT/*;
	do 
	fname=$(basename $i);
	fbname=${fname%.*};
	echo $i;
	echo $fname;
	echo "$i/$fname" " to " "$PRED_E/$fname/$fname";
	cp "$i/$fname" "$PRED_E/$fname/$fname"
done


# Prediction of relations
echo '>>>> PREDICTING RELATIONS'
python main_rels.py $PRED_E -test_model $MODEL_DIR -lowercase $lowercase -unk_token $unk_token -conflate_digits $digit_conflation -pos $pos -test_xml 0 -output_xml_dir $PRED_R -ctakes_out_dir $TEST_CTAKES_XML

