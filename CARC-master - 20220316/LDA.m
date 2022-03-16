function W = LDA(X,y)
% Usage: W = LDA(X,y)
% X: feature vectors, each row is a sample, each column is a feature
% y: label vector consists of two or more distinct labels (integers)
% W: linear discriminant vectors
% ECE539 material

    [~,d] = size(X);  % N: # of samples, d: feature dimension
    Labels = unique(y); % set of unique labels
    C = length(Labels); % # of classes
    
    %%% (a) Compute sample mean and sample covariance matrix for each class
    
    mug = mean(X);  % global mean vector  1 x d row vector
    nc = zeros(1,C);
    for i = 1:C
        id{i} = find(y == Labels(i)); % Find indices of samples in each class
        nc(i) = length(id{i});   % # samples in each class
        if nc(i) > 1
            mu{i} = mean(X(id{i},:)); % a 1 x d row vector, class mean
        else  % nc(i) == 1
            mu{i} = X(id{i},:);
            covmat{i} = zeros(d);  % degenerated case
        end
        covmat{i} = cov(X(id{i},:)); % d x d matrix
    end
    
    %%% (b) Compute within cluster scatter matrix and between class scatter
    %%% matrix
    
    SW = zeros(d); SB = zeros(d);
    for i = 1:C
        SW = SW + nc(i)*covmat{i};
        if C == 2
            SB = (mu{1} - mu{2})'*(mu{1} - mu{2});
        else  % C >=3,
            tmp = mu{i} - mug;
            SB = SB + nc(i)*tmp*tmp';
        end
    end
    
    %%% (c) Compute SVD of inv(Sw)*SB and obtain right singular vectors
    %%%     Alternatively, we solve for the generalized eigenvalue problem SB*x
    %%%     =lambda*SW*x
    
%     [W,~] = eig(SB,SW);   % W: (columns are) right eigenvectors  d x d
    [~,~,W] = svd(inv(SW)*SB);
    
    % In general, # of linear discriminant <= C - 1
    if d > C-1 
        W = W(:,1:C-1);
    end
        
end