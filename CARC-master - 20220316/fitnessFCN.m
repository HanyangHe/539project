function [sol,fitnessVal]=fitnessFCN(sol,~)%this ga is used to find the maximum value
global nPart
global pcaDim
global person0
global person1
global run
global SmallCelebrityImageData
global SmallCelebrityData
global aim
global PCA_ap
global PCA_patK
global CARC_ap
global CARC_patK

nPart=floor(sol(1)*nPartMax);
pcaDim=floor(sol(2)*pcaDimMax);
featureNumOfPerImage=size(SmallCelebrityImageData.feature,2);% the number of features for one image
partDim = floor(featureNumOfPerImage/nPart);% The number of per image feature in each PCA part
cPts = size(SmallCelebrityImageData.identity,1);%The total number of the pictures
SmallCelebrityImageData.pcaFeature = zeros(cPts, pcaDim*nPart);% Initial the size of the PCA feature matrix
changeIndex = reshape([1:featureNumOfPerImage], [], 5)';
changeIndex = changeIndex(:);% this is a special way to shuffle the index

%=============pca======================
% fprintf('Computing PCA feature...\n');
% The method divide the whole features of a image in to n part
for p = 1:nPart% execute PCA in each small part one by one
   partIndex = changeIndex([1 + (p-1)*partDim:p*partDim]);% Index sets of feature in each part for pca
   pcaIndex = [1 + (p-1)*pcaDim:p*pcaDim];% Index sets of feature after the pca process

   % PCA (only person's index number >k use PCA) without normalization
%    X = double(SmallCelebrityImageData.feature(:,partIndex));
%    [score_mappedX, PCAmapping] = pca(X(find(SmallCelebrityImageData.rank > 0), :), pcaDim);
%    X_PCA = bsxfun(@minus, X, PCAmapping.mean) * PCAmapping.M;

   % PCA (only person's index number >k use PCA) with normalization
   X = double(SmallCelebrityImageData.feature(:,partIndex));
   X=(X-ones(size(X,1),1)*mean(X))./(ones(size(X,1),1)*std(X));% N(0,1) normalization of each image feature
   [score_mappedX, PCAmapping] = pca(X(find(SmallCelebrityImageData.rank > 0), :), pcaDim);
   X_PCA = bsxfun(@minus, X, PCAmapping.mean) * PCAmapping.M;

   %original version PCA (strange data process step)
%    X = sqrt(double(SmallCelebrityImageData.feature(:,partIndex)));
%    [~, PCAmapping] = pca(X(find(SmallCelebrityImageData.rank > 0), :), pcaDim);
%    X_PCA = bsxfun(@minus, X, PCAmapping.mean) * PCAmapping.M;
%    W = diag(ones(pcaDim,1)./sqrt(PCAmapping.lambda + eps));
%    X_PCA = X_PCA*W;

   SmallCelebrityImageData.pcaFeature(:,pcaIndex) = X_PCA;%save all part pca result in sequence into pcaFeature
end

%====================LDA=========================
% ldaDim=pcaDim;
% for p = 1:nPart% execute LDA in each small part one by one
%    ldaIndex = [1 + (p-1)*ldaDim:p*ldaDim];
%    X = sqrt(double(SmallCelebrityImageData.feature(:,partIndex)));% sqrt
% 
%    [score_mappedX, PCAmapping] = myLDA(X(find(SmallCelebrityImageData.rank > 0), :), ldaDim);
%    X_PCA = ;
% 
%    SmallCelebrityImageData.pcaFeature(:,ldaIndex) = X_PCA;%save all part pca result in sequence into pcaFeature
% end


% fprintf('Computing Cross-Age Reference Coding...\n');
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
% dataset{1} = '2004-2006';
% dataset{2} = '2007-2009';
% dataset{3} = '2010-2012';

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

%Here is for prepareing your own features, the order of the features should be same as "image.list"
%{
SmallcelebrityImageData.newFeature = zeros(featureNumOfPerImage, feature_dim);
%}

% fprintf('Evaluation...\n');
queryId1 = SmallCelebrityImageData.identity(queryIndex1);
queryId2 = SmallCelebrityImageData.identity(queryIndex2);
for i = 1:3
%    fprintf(['Result for dataset ' dataset{i} '\n']);
   databaseId1 = SmallCelebrityImageData.identity(databaseIndex1{i});
   databaseId2 = SmallCelebrityImageData.identity(databaseIndex2{i});
   
   % High-Dimensional LBP method
   qX = SmallCelebrityImageData.pcaFeature(queryIndex1, :);
   X = SmallCelebrityImageData.pcaFeature(databaseIndex1{i}, :);
   dist = -1*normalizeL2(qX)*normalizeL2(X)';%normalizeL2: Eular distance standarlization of X
   %'dist' reflect a distance index of a test image to each image used as reference 
   result = evaluation(dist, queryId1, databaseId1);%mean average precision
%    fprintf('High-Dimensional LBP:\tMAP = %f, P@1 = %f\n', mean(result.ap), result.patK(1));
   PCA_ap1{run,i}=mean(result.ap);
   PCA_patK1{run,i}=result.patK(1);
   
   qX = SmallCelebrityImageData.pcaFeature(queryIndex2, :);
   X = SmallCelebrityImageData.pcaFeature(databaseIndex2{i}, :);
   dist = -1*normalizeL2(qX)*normalizeL2(X)';%normalizeL2: Eular distance standarlization of X
   %'dist' reflect a distance index of a test image to each image used as reference 
   result = evaluation(dist, queryId2, databaseId2);%mean average precision
%    fprintf('High-Dimensional LBP:\tMAP = %f, P@1 = %f\n', mean(result.ap), result.patK(1));
   PCA_ap2{run,i}=mean(result.ap);
   PCA_patK2{run,i}=result.patK(1);

   PCA_ap{run,i}=(PCA_ap1{run,i}+PCA_ap2{run,i})/2;
   PCA_patK{run,i}=(PCA_patK1{run,i}+PCA_patK2{run,i})/2;


   % CARC method
   dist = -1*normalizeL2(CARC_query1)*normalizeL2(CARC_database1{i})';
   result = evaluation(dist, queryId1, databaseId1);
%    fprintf('CARC:\t\t\tMAP = %f, P@1 = %f\n', mean(result.ap), result.patK(1));
   aim(run)=aim(run)+mean(result.ap);
   
   CARC_ap1{run,i}=mean(result.ap);
   CARC_patK1{run,i}=result.patK(1);

   dist = -1*normalizeL2(CARC_query2)*normalizeL2(CARC_database2{i})';
   result = evaluation(dist, queryId2, databaseId2);
%    fprintf('CARC:\t\t\tMAP = %f, P@1 = %f\n', mean(result.ap), result.patK(1));
   aim(run)=aim(run)+mean(result.ap);
   
   CARC_ap2{run,i}=mean(result.ap);
   CARC_patK2{run,i}=result.patK(1);

   CARC_ap{run,i}=(CARC_ap1{run,i}+CARC_ap2{run,i})/2;
   CARC_patK{run,i}=(CARC_patK1{run,i}+CARC_patK2{run,i})/2;
end
fitnessVal=aim(run);