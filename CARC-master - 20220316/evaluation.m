% Calculate mean average precision
function [result] = evaluation(dist, queryId, databaseId)
   qPts = size(queryId,1);%test set
   nPts = size(databaseId,1);%train set
   totalK = 20;%original is 20
   ap = zeros(qPts, 1);%test set
   patK = zeros(qPts, totalK);%test set totalK turns
   rankResults = zeros(qPts, nPts);
   for i = 1:qPts% pick test set image by turn
      [~, idx] = sort(dist(i,:), 'ascend');%find the column index vector in the 'dist' increase sequence for row-i test image feature
      % The idx vector is the index/rank of the databaseId from smallest distance to biggest distance

      correctRank = find(databaseId(idx) == queryId(i));%find the index(rank) of the right ID in datebaseID vector
      % only the index/rank in correctRank are correct result in the identification job, others are failed/wrong

      rankResults(i,:) = idx;% save idx in each row of each image

      nAns = size(correctRank,1);
      for j = 1:nAns% select correctRank by turn
         ap(i) = ap(i) + j/correctRank(j);%this is accurency/efficiency, fully marks is 1, only the distance of all correct classify result ranks 
         % have smaller value than other wrong result distance can get
         % fully marks.
      end
      for k = 1:totalK
         patK(i,k) = sum(databaseId(idx(1:k)) == queryId(i))/k;%cheak how many identify results are correct in first totalK smallest distance image
         % the return is the average value of the correct image identify number
      end
      ap(i) = ap(i)/nAns;%average of ap(i)
   end
   result.ap = ap;
   result.patK = mean(patK);
   result.rankResults = rankResults;