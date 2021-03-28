# Pipeline helper for setting up data transfer



######################################
# Pipeline doc:
# 
# Load initial batch:
# 	Dataset = (Train[16],Validation[8])
# 	LoadRandData(): Return Dataset
# 
# -MainCycle-
#
# Aync Start Load Next Data Batch:
# 	Loaded=FALSE
# 	Async(LoadRandData()=>NextDataBatch):
# 		OnComplete: Set Loaded=TRUE
# 	Start Fit:
#		VAE.fit(Train,Validation)
# 
# If Loaded = FALSE:
# 	Await Loaded = TRUE
# 
# Dataset = NextDataBatch
# Loop MainCycle
#
######################################

