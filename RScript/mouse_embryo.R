#Load Library
library(Matrix)

U = readRDS("/nfs/turbo/umms-welchjd/yichen/data/scRNA/MOCA_intron.rds")
S = readRDS("/nfs/turbo/umms-welchjd/yichen/data/scRNA/MOCA_exon.rds")
anno = readRDS("/nfs/turbo/umms-welchjd/yichen/data/scRNA/MOCA_df_cell.rds")

X<-S@x+U@x

#Create dataframes
#unspliced
row <- U@i
val <- U@x
col <- U@p
dfs <- data.frame(row, val)
write.csv(dfs,'/scratch/blaauw_root/blaauw1/gyichen/U_ix.csv', row.names = FALSE)
dfs <- data.frame(col)
write.csv(dfs,'/scratch/blaauw_root/blaauw1/gyichen/U_j.csv', row.names = FALSE)


#spliced
row <- S@i
val <- S@x
col <- S@p
dfs <- data.frame(row, val)
write.csv(dfs,'/scratch/blaauw_root/blaauw1/gyichen/S_ix.csv', row.names = FALSE)
dfs <- data.frame(col)
write.csv(dfs,'/scratch/blaauw_root/blaauw1/gyichen/S_j.csv', row.names = FALSE)



genes <- S@Dimnames[[1]]
cells <- S@Dimnames[[2]]
dfg <- data.frame(genes)
write.csv(dfg,'/scratch/blaauw_root/blaauw1/gyichen/gene_name.csv', row.names = FALSE)
dfc <- data.frame(cells)
write.csv(dfc,'/scratch/blaauw_root/blaauw1/gyichen/cell_name.csv', row.names = FALSE)

summary(U@x)
