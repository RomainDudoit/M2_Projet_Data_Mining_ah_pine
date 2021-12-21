setwd("D:/DIAPO/App supervisé et non supervisé/td4")

#Lecture et description des donnees
#2
D = read.table("breast-cancer-wisconsin.data", sep = ",", na.strings = "?")
colnames(D) = c("code_number", "clump_thickness", "cell_size_uni", "cell_shape_uni", "marginal_adhesion", "single_epithetial_cell_size", "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitoses", "class")

#3
class(D) #df
str(D) #nbr obs/var et type de chaque var
head(D) #les premieres lignes
summary(D) #resume de chaque var

#Separation des donnees en 'train' et 'test'

#4 
na_row = complete.cases(D)[complete.cases(D)==FALSE]  
length(na_row) #on a bien 16 lignes non completes

#5 Garder les donnees complete
D = D[complete.cases(D),]
nrow(D) #on a bien 699-16 = 683 lignes

#6 Cr?ation des variable X et Y :
X = as.matrix(D[, 2:10]) #donnees explicatives
y = D$class #variable cible

#7 Recodage de Y :
library(dplyr)
#Distribution de y
length(y[y==2]) #444
length(y[y==4]) #239
y = recode(y, "2" = 0, "4" = 1)
#Verification de la distribution de y apres recodage :
length(y[y==0]) #444 -> benin
length(y[y==1]) #239 -> maligne
#on a bien recode

#8 D?coupage des variables benin et malin
benin = which(y == 0, arr.ind = TRUE)
#length(benin)
malin = which(y == 1, arr.ind = TRUE)
#length(malin)


#9 Selection des 200 observations begnines
train_set = benin[1:200]
test_set = -benin[1:200] 

Xtrain_set = X[train_set,]
Xtest_set = X[test_set,]
ytrain_set = y[train_set]
ytest_set = y[test_set]

#One-class SVM
#10 Chargement de la library
library(e1071)

#11 Estimation du modele avec noyau gaussien et de type "one-classification"
oc_svm_fit = svm(as.matrix(Xtrain_set), type = "one-classification", gamma = 1/2)


#12 Score des observations de test :
oc_svm_pred_test = predict(oc_svm_fit, newdata = Xtest_set, decision.values = TRUE)
str(oc_svm_pred_test) # voir ses attributs


#13
attr(oc_svm_pred_test, "decision.values") 
#on obtient les scores de chaque observation
oc_svm_score_test = -as.numeric(attr(oc_svm_pred_test ,"decision.values")) 
#on a changer le signe 
oc_svm_score_test
min( oc_svm_score_test)
max( oc_svm_score_test)

#Courbe ROC
#14 Chargement library
library(ROCR)

#15 
pred_oc_svm = prediction(oc_svm_score_test, y[test_set])
#transforme le modele dans le format utilisÃ© par ROCR
oc_svm_roc = performance(pred_oc_svm, measure = "tpr", x.measure = "fpr")
#str(oc_svm_roc)
#oc_svm_roc@alpha.values
#On obtient un obet de classe performance composÃ© des mesures testÃ©es 
#(taux de vrais positifs et le taux de faux positifs) pour diffÃ©rentes valeurs de alpha
plot(oc_svm_roc)

#16 
#Il semblerait que le classifieur crÃ©Ã© soit assez performant puisque
#1- il est au dessus du classifieur alÃ©atoire
#2- le meilleur 'point' est assez proche du point idÃ©al (0,1)

oc_svm_auc <- performance(pred_oc_svm, "auc")
oc_svm_auc@y.values[[1]]
#L'aire sour la courbe (AUC) est de 0.993, ce qui est tres proche 
#de 1 (AUC du meilleur modele possible). Notre modele est donc trÃ¨s bon.


#6 Kernel PCA
library(kernlab)
kernel = rbfdot(sigma = 1/8) 
#crÃ©ation objet de type kernel
Ktrain = kernelMatrix(kernel, X[train_set,])
#Calcule la matrice K sur les donnees d'apprentissage
class(Ktrain)

#18
n = nrow(Ktrain)
k2 = apply(Ktrain, 1, sum) #somme en ligne
k3 = apply(Ktrain, 2, sum) #somme en colonne
k4 = sum (Ktrain) #somme de tous les elements


# (vecteurs centres dans F) :
#n correspond au nombre de lignes de l'ensemble d'apprentissage
KtrainCent = matrix (0, ncol = n, nrow = n) 
# On initialise une matrice avec des zeros
for(i in 1:n){
  for(j in 1:n){
    KtrainCent[i, j] = Ktrain[i, j] - 1/n*k2[i] - 1/n*k3[j] + 1/n^2*k4
  }
}
# Le code dans cette boucle transforme la matrice Ã  noyau K 
# en un produit scalaire des vecteurs centres dans F, en utilisant la formule 1 :
#On retrouve bien : k2 = la premiÃ¨re somme
#k3 = la seconde somme
#k4 = la double somme


#19 Decomposition spectrale : 
eigen_KtrainCent = eigen(KtrainCent)


#20 Calcul des coefiicients alpha
s = 80
#s est le nombre d'axes principaux gardes
A = eigen_KtrainCent$vectors[, 1:s]%*%diag(1/sqrt(eigen_KtrainCent$values[1:s]))
A
#αm = 1/√λm * v^m
#coeff alpha


#21 noyau sur l'echantillon total :
#X est l'echantillon total
K = kernelMatrix(kernel, X)
#K[train_set,train_set]==Ktrain


dim(K) # Matrice noyau K 
K[3,3]
#Données test : 
K_test = K[test_set,test_set]
dim(K_test)

K_test[1,1]

#22 calcule du carre de la distance euclidienne entre l'origine et le vecteur
#KTrain d'en haut devient K :
n=683
#p1 = k(z,z)
p1=as.numeric(diag(K[test_set,test_set]))
p2 =apply(K[test_set,train_set],1,sum)
p2
p3=sum(Ktrain)
p3


#23 : ps
ps=p1-(2/n *p2)+(1/n^2 *p3)

str(ps)

#ps est de bonnes dimensions

#24 Terme de (5) : 

f1 = K[test_set,train_set]
dim(f1)

f2 = apply(K[train_set,train_set], 1, sum)
f2

f3 = apply(K[test_set,train_set], 1, sum)
f3

f4 = sum(K[train_set,train_set])
f4

#25

intermediaire = f1 - (1/n * f2) - (1/n * f3) + (1/n^2 * f4)

dim(A)
dim(intermediaire)

# somme alpha 
f = intermediaire %*%  A
dim(f)
#les dimensions sont ok 

#26
eq6 = apply(f^2,1,sum)
str(eq6)

kpca_score_test = ps - eq6
str(kpca_score_test)
kpca_score_test


#27

pred_oc_kpca = prediction(kpca_score_test,y[test_set])
oc_kpca_roc = performance(pred_oc_kpca, measure = "tpr", x.measure= "fpr")
#pdf('rplot.pdf')
plot(oc_svm_roc)
plot(oc_kpca_roc, add=TRUE, col=2)
legend(x = "bottomright", 
       legend = c("One-class SVM", "ACP Kernel"),
       fill = 1:2)

#calculer auc
auc_ROCR <- performance(pred_oc_kpca, measure = "auc")
auc_ROCR <- auc_ROCR@y.values[[1]]
print(auc_ROCR)

