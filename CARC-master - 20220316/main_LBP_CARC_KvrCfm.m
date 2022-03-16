clear
clc
global nPart
global pcaDim
global person0
global person1
% In this code, the author provide a new feature process link after PCA to
% improve the identification performance
fprintf('Loading dataset...\n');
load('SmallCelebrityData.mat');
load('SmallCelebrityImageData.mat');


eps = 10^-5;

aim=[];
aim(1)=0;

nPart(1)=16;% The part number of PCA (default=16)
pcaDim(1)=500;% The dimension of PCA (default=500)

step=[];
theta=[];
step(1)=5;
theta(1)=0;%rad

person0=1;%left boundary of person index used in the experiment
person1=10;%right boundary of person index used in the experiment

K=5;% view range of nearest classifer (do classify based on K nearest number of train sample)

%==========================================================================

featureNumOfPerImage=size(SmallCelebrityImageData.feature,2);% the number of features for one image
partDim = floor(featureNumOfPerImage/nPart);% The number of per image feature in each PCA part
cPts = size(SmallCelebrityImageData.identity,1);%The total number of the pictures
SmallCelebrityImageData.pcaFeature = zeros(cPts, pcaDim*nPart);% Initial the size of the PCA feature matrix
changeIndex = reshape([1:featureNumOfPerImage], [], 5)';
changeIndex = changeIndex(:);% this is a special way to shuffle the index

fprintf('Computing PCA feature...\n');
%=============pca======================
% The method divide the whole features of a image in to n part
for p = 1:nPart% execute PCA in each small part one by one
   partIndex = changeIndex([1 + (p-1)*partDim:p*partDim]);% Index sets of feature in each part for pca
   pcaIndex = [1 + (p-1)*pcaDim:p*pcaDim];% Index sets of feature after the pca process

   % PCA (only person's index number >k use PCA) with normalization
   X = double(SmallCelebrityImageData.feature(:,partIndex));
   X=(X-ones(size(X,1),1)*mean(X))./(ones(size(X,1),1)*std(X));% N(0,1) normalization of each image feature
   [score_mappedX, PCAmapping] = pca(X, pcaDim);
   X_PCA = bsxfun(@minus, X, PCAmapping.mean) * PCAmapping.M;

   SmallCelebrityImageData.pcaFeature(:,pcaIndex) = X_PCA;%save all part pca result in sequence into pcaFeature
end


fprintf('Computing Cross-Age Reference Coding...\n');
%choose 2004-2012 3 year-layer-devided people who's indexes are person0~person1 as train set
databaseIndex1{1} = find((SmallCelebrityImageData.year == 2004 | SmallCelebrityImageData.year == 2005 | SmallCelebrityImageData.year == 2006) & SmallCelebrityImageData.rank <=person1 & SmallCelebrityImageData.rank >= person0);
databaseIndex1{2} = find((SmallCelebrityImageData.year == 2007 | SmallCelebrityImageData.year == 2008 | SmallCelebrityImageData.year == 2009) & SmallCelebrityImageData.rank <=person1 & SmallCelebrityImageData.rank >= person0);
databaseIndex1{3} = find((SmallCelebrityImageData.year == 2010 | SmallCelebrityImageData.year == 2011 | SmallCelebrityImageData.year == 2012) & SmallCelebrityImageData.rank <=person1 & SmallCelebrityImageData.rank >= person0);
%choose 2013 people who's index are person0~person1 as a test set
queryIndex1 = find(SmallCelebrityImageData.year == 2013 & SmallCelebrityImageData.rank <=person1 & SmallCelebrityImageData.rank >= person0);

% ============================feature transform===================================
lambda = 10;
lambda2 = 10000;
CARC_query1 = CARC(SmallCelebrityImageData, SmallCelebrityData, lambda, lambda2, queryIndex1);
CARC_database1{1} = CARC(SmallCelebrityImageData, SmallCelebrityData, lambda, lambda2, databaseIndex1{1});
CARC_database1{2} = CARC(SmallCelebrityImageData, SmallCelebrityData, lambda, lambda2, databaseIndex1{2});
CARC_database1{3} = CARC(SmallCelebrityImageData, SmallCelebrityData, lambda, lambda2, databaseIndex1{3});
% =========================================================================

dataset1{1} = '2004-2006';
dataset1{2} = '2007-2009';
dataset1{3} = '2010-2012';

%choose 2005-2013 3 year-layer-devided people who's indexes are person0~person1 as train set
databaseIndex2{1} = find((SmallCelebrityImageData.year == 2005 | SmallCelebrityImageData.year == 2006 | SmallCelebrityImageData.year == 2007) & SmallCelebrityImageData.rank <=person1 & SmallCelebrityImageData.rank >= person0);
databaseIndex2{2} = find((SmallCelebrityImageData.year == 2008 | SmallCelebrityImageData.year == 2009 | SmallCelebrityImageData.year == 2010) & SmallCelebrityImageData.rank <=person1 & SmallCelebrityImageData.rank >= person0);
databaseIndex2{3} = find((SmallCelebrityImageData.year == 2011 | SmallCelebrityImageData.year == 2012 | SmallCelebrityImageData.year == 2013) & SmallCelebrityImageData.rank <=person1 & SmallCelebrityImageData.rank >= person0);
%choose 2004 people who's index are person0~person1 as a test set
queryIndex2 = find(SmallCelebrityImageData.year == 2004 & SmallCelebrityImageData.rank <=person1 & SmallCelebrityImageData.rank >= person0);

% ============================feature transform===================================
lambda = 10;
lambda2 = 10000;
CARC_query2 = CARC(SmallCelebrityImageData, SmallCelebrityData, lambda, lambda2, queryIndex2);
CARC_database2{1} = CARC(SmallCelebrityImageData, SmallCelebrityData, lambda, lambda2, databaseIndex2{1});
CARC_database2{2} = CARC(SmallCelebrityImageData, SmallCelebrityData, lambda, lambda2, databaseIndex2{2});
CARC_database2{3} = CARC(SmallCelebrityImageData, SmallCelebrityData, lambda, lambda2, databaseIndex2{3});
% =========================================================================

dataset2{1} = '2005-2007';
dataset2{2} = '2008-2010';
dataset2{3} = '2011-2013';

%Here is for prepareing your own features, the order of the features should be same as "image.list"
%{
SmallcelebrityImageData.newFeature = zeros(featureNumOfPerImage, feature_dim);
%}
MAP_LBP=0;
MAP_CARC=0;
P1_LBP=0;
P1_CARC=0;
fprintf('Evaluation...\n');
queryId1 = SmallCelebrityImageData.identity(queryIndex1);
disp(['Choose 2013 as test set:'])
for i = 1:3
   fprintf(['Result for dataset ' dataset1{i} '\n']);
   databaseId1 = SmallCelebrityImageData.identity(databaseIndex1{i});
   
   % High-Dimensional LBP method=====================
   qX = SmallCelebrityImageData.pcaFeature(queryIndex1, :);
   X = SmallCelebrityImageData.pcaFeature(databaseIndex1{i}, :);
   dist = -1*normalizeL2(qX)*normalizeL2(X)';%normalizeL2: Eular distance standarlization of X
%    dist=EularDistOfFeatureMat(qX,X);
   %'dist' reflect a distance index of a test image to each image used as reference 
   result = evaluation(dist, queryId1, databaseId1);%mean average precision
   fprintf('High-Dimensional LBP:\tMAP = %f, P@1 = %f\n', mean(result.ap), result.patK(1));
   MAP_LBP=MAP_LBP+mean(result.ap);
   P1_LBP=P1_LBP+result.patK(1);

   % K view range classifier
   preQueryId1_LBP{i}=KrangeDistClassifier(dist,K, databaseId1);
   C1_LBP{i} = confusionmat(queryId1,preQueryId1_LBP{i});
%    disp(['confusion matrix of LBP: '])
%    C1_LBP{i}

   % CARC method===========================================================
   dist = -1*normalizeL2(CARC_query1)*normalizeL2(CARC_database1{i})';
   % CARC_query1 is a membership/relationship matrix between test vector and feature,
   % CARC_database1{i} is a membership/relationship matrix between train vector and feature
   % CARC_query1*CARC_database1{i} is a membership/relationship matrix between test
   % vector and train vector (this step approximate the synthesis of fuzzy relations using matrix multiplication)
   % add a minus to change membership to distance

%    dist=EularDistOfFeatureMat(CARC_query1,CARC_database1{i});
   result = evaluation(dist, queryId1, databaseId1);
   fprintf('LBP+CARC:\t\t\tMAP = %f, P@1 = %f\n', mean(result.ap), result.patK(1));

    % K view range classifier
   preQueryId1_CARC{i}=KrangeDistClassifier(dist,K, databaseId1);
   C1_CARC{i} = confusionmat(queryId1,preQueryId1_CARC{i});
%    disp(['confusion matrix of LBP: '])
%    C1_CARC{i}

   MAP_CARC=MAP_CARC+mean(result.ap);
   P1_CARC=P1_CARC+result.patK(1);
end  
disp([' '])
disp(['Choose 2004 as test set:'])
for i = 1:3
   queryId2 = SmallCelebrityImageData.identity(queryIndex2);

   fprintf(['Result for dataset ' dataset2{i} '\n']);
   databaseId2 = SmallCelebrityImageData.identity(databaseIndex2{i});
   
   % basic High-Dimensional LBP method
   qX = SmallCelebrityImageData.pcaFeature(queryIndex2, :);
   X = SmallCelebrityImageData.pcaFeature(databaseIndex2{i}, :);
   dist = -1*normalizeL2(qX)*normalizeL2(X)';%normalizeL2: Eular distance standarlization of X
%    dist=EularDistOfFeatureMat(qX,X);
   %'dist' reflect a distance index of a test image to each image used as reference 
   result = evaluation(dist, queryId2, databaseId2);%mean average precision
   fprintf('High-Dimensional LBP:\tMAP = %f, P@1 = %f\n', mean(result.ap), result.patK(1));
   MAP_LBP=MAP_LBP+mean(result.ap);
   P1_LBP=P1_LBP+result.patK(1);

   % K view range classifier
   preQueryId2_LBP{i}=KrangeDistClassifier(dist,K, databaseId2);
   C2_LBP{i} = confusionmat(queryId2,preQueryId2_LBP{i});
%    disp(['confusion matrix of LBP: '])
%    C2_LBP{i}

   % CARC method===========================================================
   dist = -1*normalizeL2(CARC_query2)*normalizeL2(CARC_database2{i})';
%    dist=EularDistOfFeatureMat(CARC_query1,CARC_database1{i});
   result = evaluation(dist, queryId2, databaseId2);
   fprintf('LBP+CARC:\t\t\tMAP = %f, P@1 = %f\n', mean(result.ap), result.patK(1));

    % K view range classifier
   preQueryId2_CARC{i}=KrangeDistClassifier(dist,K, databaseId2);
   C2_CARC{i} = confusionmat(queryId2,preQueryId2_CARC{i});
%    disp(['confusion matrix of LBP: '])
%    C2_CARC{i}

   MAP_CARC=MAP_CARC+mean(result.ap);
   P1_CARC=P1_CARC+result.patK(1);
end

MAP_LBP=MAP_LBP/6;
MAP_CARC=MAP_CARC/6;
P1_LBP=P1_LBP/6;
P1_CARC=P1_CARC/6;
C_LBP=C1_LBP{1}+C1_LBP{2}+C1_LBP{3}+C2_LBP{1}+C2_LBP{2}+C2_LBP{3};
C_CARC=C1_CARC{1}+C1_CARC{2}+C1_CARC{3}+C2_CARC{1}+C2_CARC{2}+C2_CARC{3};
Pr_LBP=sum(diag(C_LBP))/sum(sum(C_LBP));
Pr_CARC=sum(diag(C_CARC))/sum(sum(C_CARC));
disp([' '])
disp(['Summary results of naive LBP:'])
fprintf('High-Dimensional LBP:\tMAP_ave = %f, P@1_ave = %f\n', MAP_LBP,P1_LBP);
disp(['Confusion Matrix of LBP:'])
C_LBP
disp(['Pr. Classification = ' num2str(Pr_LBP*100), '%'])

disp([' '])
disp(['Summary results of LBP+CARC:'])
fprintf('CARC:\t\t\tMAP_ave = %f, P@1_ave = %f\n', MAP_CARC,P1_CARC);
disp(['Confusion Matrix of CARC:'])
C_CARC
disp(['Pr. Classification = ' num2str(Pr_CARC*100), '%'])