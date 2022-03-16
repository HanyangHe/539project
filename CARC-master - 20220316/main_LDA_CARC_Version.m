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

fprintf('Computing LDA feature...\n');
eps = 10^-5;

aim=[];
aim(1)=0;

nPart(1)=8;% The part number of PCA (default=16)
% pcaDim(1)=500;% The dimension of PCA (default=500)

step=[];
theta=[];
step(1)=5;
theta(1)=0;%rad

person0=1;%left boundary of person index used in the experiment
person1=10;%right boundary of person index used in the experiment

featureNumOfPerImage=size(SmallCelebrityImageData.feature,2);% the number of features for one image
partDim = floor(featureNumOfPerImage/nPart);% The number of per image feature in each PCA part
cPts = size(SmallCelebrityImageData.identity,1);%The total number of the pictures
% SmallCelebrityImageData.pcaFeature = zeros(cPts, pcaDim*nPart);% Initial the size of the PCA feature matrix
changeIndex = reshape([1:featureNumOfPerImage], [], 5)';
changeIndex = changeIndex(:);% this is a special way to shuffle the index

%=============LDA======================
% The method divide the whole features of a image in to n part
for p = 1:nPart% execute LDA in each small part one by one
   partIndex = changeIndex([1 + (p-1)*partDim:p*partDim]);% Index sets of feature in each part for pca
%    pcaIndex = [1 + (p-1)*pcaDim:p*pcaDim];% Index sets of feature after the pca process


   % LDA (only person's index number >k use LDA) with normalization
   X = double(SmallCelebrityImageData.feature(:,partIndex));
   y=double(SmallCelebrityImageData.identity(:,1));
   X=(X-ones(size(X,1),1)*mean(X))./(ones(size(X,1),1)*std(X));% N(0,1) normalization of each image feature
   [wLDA] = LDA(X, y);
   X_LDA = X*wLDA;
   LDA_Dim=size(X_LDA,2);
   LDA_Index = [1 + (p-1)*LDA_Dim:p*LDA_Dim];

   SmallCelebrityImageData.pcaFeature(:,LDA_Index) = X_LDA;%save all part pca result in sequence into pcaFeature
end
pcaDim=LDA_Dim;%for unifing name used in CARC



fprintf('Computing Cross-Age Reference Coding...\n');
%choose 2004-2012 3 year-layer-devided people who's indexes are person0~person1 as train set
databaseIndex1{1} = find((SmallCelebrityImageData.year == 2004 | SmallCelebrityImageData.year == 2005 | SmallCelebrityImageData.year == 2006) & SmallCelebrityImageData.rank <=person1 & SmallCelebrityImageData.rank >= person0);
databaseIndex1{2} = find((SmallCelebrityImageData.year == 2007 | SmallCelebrityImageData.year == 2008 | SmallCelebrityImageData.year == 2009) & SmallCelebrityImageData.rank <=person1 & SmallCelebrityImageData.rank >= person0);
databaseIndex1{3} = find((SmallCelebrityImageData.year == 2010 | SmallCelebrityImageData.year == 2011 | SmallCelebrityImageData.year == 2012) & SmallCelebrityImageData.rank <=person1 & SmallCelebrityImageData.rank >= person0);
%choose 2013 people who's index are person0~person1 as a test set
queryIndex1 = find(SmallCelebrityImageData.year == 2013 & SmallCelebrityImageData.rank <=person1 & SmallCelebrityImageData.rank >= person0);
lambda = 10;
lambda2 = 10000;
CARC_query1 = CARC(SmallCelebrityImageData, SmallCelebrityData, lambda, lambda2, queryIndex1);
CARC_database1{1} = CARC(SmallCelebrityImageData, SmallCelebrityData, lambda, lambda2, databaseIndex1{1});
CARC_database1{2} = CARC(SmallCelebrityImageData, SmallCelebrityData, lambda, lambda2, databaseIndex1{2});
CARC_database1{3} = CARC(SmallCelebrityImageData, SmallCelebrityData, lambda, lambda2, databaseIndex1{3});
dataset1{1} = '2004-2006';
dataset1{2} = '2007-2009';
dataset1{3} = '2010-2012';

%choose 2005-2013 3 year-layer-devided people who's indexes are person0~person1 as train set
databaseIndex2{1} = find((SmallCelebrityImageData.year == 2005 | SmallCelebrityImageData.year == 2006 | SmallCelebrityImageData.year == 2007) & SmallCelebrityImageData.rank <=person1 & SmallCelebrityImageData.rank >= person0);
databaseIndex2{2} = find((SmallCelebrityImageData.year == 2008 | SmallCelebrityImageData.year == 2009 | SmallCelebrityImageData.year == 2010) & SmallCelebrityImageData.rank <=person1 & SmallCelebrityImageData.rank >= person0);
databaseIndex2{3} = find((SmallCelebrityImageData.year == 2011 | SmallCelebrityImageData.year == 2012 | SmallCelebrityImageData.year == 2013) & SmallCelebrityImageData.rank <=person1 & SmallCelebrityImageData.rank >= person0);
%choose 2004 people who's index are person0~person1 as a test set
queryIndex2 = find(SmallCelebrityImageData.year == 2004 & SmallCelebrityImageData.rank <=person1 & SmallCelebrityImageData.rank >= person0);
lambda = 10;
lambda2 = 10000;
CARC_query2 = CARC(SmallCelebrityImageData, SmallCelebrityData, lambda, lambda2, queryIndex2);
CARC_database2{1} = CARC(SmallCelebrityImageData, SmallCelebrityData, lambda, lambda2, databaseIndex2{1});
CARC_database2{2} = CARC(SmallCelebrityImageData, SmallCelebrityData, lambda, lambda2, databaseIndex2{2});
CARC_database2{3} = CARC(SmallCelebrityImageData, SmallCelebrityData, lambda, lambda2, databaseIndex2{3});
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
for i = 1:3
   fprintf(['Result for dataset ' dataset1{i} '\n']);
   databaseId1 = SmallCelebrityImageData.identity(databaseIndex1{i});
   
   % High-Dimensional LBP method
   qX = SmallCelebrityImageData.pcaFeature(queryIndex1, :);
   X = SmallCelebrityImageData.pcaFeature(databaseIndex1{i}, :);
   dist = -1*normalizeL2(qX)*normalizeL2(X)';%normalizeL2: Eular distance standarlization of X
   %'dist' reflect a distance index of a test image to each image used as reference 
   result = evaluation(dist, queryId1, databaseId1);%mean average precision
   fprintf('High-Dimensional LBP:\tMAP = %f, P@1 = %f\n', mean(result.ap), result.patK(1));
   MAP_LBP=MAP_LBP+mean(result.ap);
   P1_LBP=P1_LBP+result.patK(1);

   % CARC method
   dist = -1*normalizeL2(CARC_query1)*normalizeL2(CARC_database1{i})';
   result = evaluation(dist, queryId1, databaseId1);
   fprintf('CARC:\t\t\tMAP = %f, P@1 = %f\n', mean(result.ap), result.patK(1));
   MAP_CARC=MAP_CARC+mean(result.ap);
   P1_CARC=P1_CARC+result.patK(1);
end  
for i = 1:3  
   queryId2 = SmallCelebrityImageData.identity(queryIndex2);

   fprintf(['Result for dataset ' dataset2{i} '\n']);
   databaseId2 = SmallCelebrityImageData.identity(databaseIndex2{i});
   
   % High-Dimensional LBP method
   qX = SmallCelebrityImageData.pcaFeature(queryIndex2, :);
   X = SmallCelebrityImageData.pcaFeature(databaseIndex2{i}, :);
   dist = -1*normalizeL2(qX)*normalizeL2(X)';%normalizeL2: Eular distance standarlization of X
   %'dist' reflect a distance index of a test image to each image used as reference 
   result = evaluation(dist, queryId2, databaseId2);%mean average precision
   fprintf('High-Dimensional LBP:\tMAP = %f, P@1 = %f\n', mean(result.ap), result.patK(1));
   MAP_LBP=MAP_LBP+mean(result.ap);
   P1_LBP=P1_LBP+result.patK(1);

   % CARC method
   dist = -1*normalizeL2(CARC_query2)*normalizeL2(CARC_database2{i})';
   result = evaluation(dist, queryId2, databaseId2);
   fprintf('CARC:\t\t\tMAP = %f, P@1 = %f\n', mean(result.ap), result.patK(1));

   MAP_CARC=MAP_CARC+mean(result.ap);
   P1_CARC=P1_CARC+result.patK(1);
end

MAP_LBP=MAP_LBP/6;
MAP_CARC=MAP_CARC/6;
P1_LBP=P1_LBP/6;
P1_CARC=P1_CARC/6;

fprintf('High-Dimensional LBP:\tMAP_ave = %f, P@1_ave = %f\n', MAP_LBP,P1_LBP);
fprintf('CARC:\t\t\tMAP_ave = %f, P@1_ave = %f\n', MAP_CARC,P1_CARC);