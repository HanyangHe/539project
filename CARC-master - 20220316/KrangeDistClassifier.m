function preClassLabel=KrangeDistClassifier(dist,K, databaseId)
% dist: is the distance matrix between test set and train set (a row is a test object, a column is a train object)
% K: the number of the nearist points used to make classify
% databaseId: training set ID
% classLabel: the final determined result

qPts = size(dist,1);%test sample num
nPts = size(dist,2);%train sample num
rankResults = zeros(qPts, nPts);
for i = 1:qPts% run turn of each test object
    [~, idx] = sort(dist(i,:), 'ascend');%find the column index vector in the 'dist' increase sequence for row-i test image feature
      % The idx vector is the index/rank of the databaseId from smallest distance to biggest distance
    rankResults(i,:) = idx;% save idx in each row of each image
end
rankInView=rankResults(:,1:K);
for i = 1:qPts
    for j=1:K
        Result(i,j)=databaseId(rankInView(i,j));
    end
end
preClassLabel=mode(Result,2);