function [idx, dist] = knnsearch(X,Y,varargin)
%KNNSEARCH Find K nearest neighbors.
%   IDX = KNNSEARCH(X,Y) finds the nearest neighbor in X for each point in
%   Y. X is an MX-by-N matrix and Y is an MY-by-N matrix. Rows of X and Y
%   correspond to observations and columns correspond to variables. IDX is
%   a column vector with MY rows. Each row in IDX contains the index of
%   the nearest neighbor in X for the corresponding row in Y.
%
%   [IDX, D] = KNNSEARCH(X,Y) returns a MY-by-1 vector D containing the
%   distances between each row of Y and its closest point in X.
%
%   [IDX, D]= KNNSEARCH(X,Y,'NAME1',VALUE1,...,'NAMEN',VALUEN) specifies
%   optional argument name/value pairs:
%
%     Name          Value
%     'K'           A positive integer, K, specifying the number of nearest
%                   neighbors in X to find for each point in Y. Default is
%                   1. IDX and D are MY-by-K matrices. D sorts the
%                   distances in each row in ascending order. Each row in
%                   IDX contains the indices of K closest neighbors in X
%                   corresponding to the K smallest distances in D.
%
%     'NSMethod'    Nearest neighbors search method. Value is either:
%                   'kdtree'    - Creates and uses a kd-tree to find
%                                 nearest neighbors. 'kdtree' is only valid
%                                 when the distance metric is one of the
%                                 following metrics:
%                                   - 'euclidean'
%                                   - 'cityblock'
%                                   - 'minkowski'
%                                   - 'chebychev'
%                   'exhaustive' - Uses the exhaustive search algorithm.
%                                  The distance values from all the points
%                                  in X to each point in Y are computed to
%                                  find nearest neighbors.
%                   Default is 'kdtree' when the number of columns of X is
%                   not greater than 10, X is not sparse, and the distance
%                   metric is one of the above 4 metrics; otherwise,
%                   default is 'exhaustive'.
%
%    'IncludeTies'  A logical value indicating whether KNNSEARCH will
%                   include all the neighbors whose distance values are
%                   equal to the Kth smallest distance. Default is false.
%                   If the value is true, KNNSEARCH includes all these
%                   neighbors. In this case, IDX and D are MY-by-1 cell
%                   arrays. Each row in IDX and D contains a vector with at
%                   least K numeric numbers. D sorts the distances in each
%                   vector in ascending order. Each row in IDX contains the
%                   indices of the closest neighbors corresponding to these
%                   smallest distances in D.
%
%    'Distance'     A string or a function handle specifying the distance
%                   metric. The value can be one of the following:
%                   'euclidean'   - Euclidean distance (default).
%                   'seuclidean'  - Standardized Euclidean distance. Each
%                                   coordinate difference between X and a
%                                   query point is scaled by dividing by a
%                                   scale value S. The default value of S
%                                   is the standard deviation computed from
%                                   X, S=NANSTD(X). To specify another
%                                   value for S, use the 'Scale' argument.
%                   'fasteuclidean'    
%                                 - Euclidean distance computed by using an
%                                   alternative algorithm that saves time. 
%                                   This faster algorithm can, in some 
%                                   cases, reduce accuracy.
%                   'fastseuclidean'   
%                                 - Standardized Euclidean distance computed 
%                                   by using an alternative algorithm that 
%                                   saves time. This faster algorithm can, 
%                                   in some cases, reduce accuracy.
%                   'cityblock'   - City Block distance.
%                   'chebychev'   - Chebychev distance (maximum coordinate
%                                   difference).
%                   'minkowski'   - Minkowski distance. The default
%                                   exponent is 2. To specify a different
%                                   exponent, use the 'P' argument.
%                   'mahalanobis' - Mahalanobis distance, computed using a
%                                   positive definite covariance matrix C.
%                                   The default value of C is the sample
%                                   covariance matrix of X, as computed by
%                                   NANCOV(X). To specify another value for
%                                   C, use the 'Cov' argument.
%                   'cosine'      - One minus the cosine of the included
%                                   angle between observations (treated as
%                                   vectors).
%                   'correlation' - One minus the sample linear
%                                   correlation between observations
%                                   (treated as sequences of values).
%                   'spearman'    - One minus the sample Spearman's rank
%                                   correlation between observations
%                                  (treated as sequences of values).
%                   'hamming'     - Hamming distance, percentage of
%                                   coordinates that differ.
%                   'jaccard'     - One minus the Jaccard coefficient, the
%                                   percentage of nonzero coordinates that
%                                   differ.
%                   function      - A distance function specified using @
%                                   (for example @DISTFUN). A distance
%                                   function must be of the form
%  
%                                   function D2 = DISTFUN(ZI, ZJ),
%  
%                                   taking as arguments a 1-by-N vector ZI
%                                   containing a single row of X or Y, an
%                                   M2-by-N matrix ZJ containing multiple
%                                   rows of X or Y, and returning an
%                                   M2-by-1 vector of distances D2, whose
%                                   Jth element is the distance between the
%                                   observations ZI and ZJ(J,:).
%
%    'P'            A positive scalar indicating the exponent of Minkowski
%                   distance. This argument is only valid when 'Distance'
%                   is 'minkowski'. Default is 2.
%  
%    'Cov'          A positive definite matrix indicating the covariance
%                   matrix when computing the Mahalanobis distance. This
%                   argument is only valid when 'Distance' is
%                  'mahalanobis'. Default is NANCOV(X).
%  
%    'Scale'        A vector S containing non-negative values, with length
%                   equal to the number of columns in X. Each coordinate
%                   difference between X and a query point is scaled by the
%                   corresponding element of S. This argument is only valid
%                   when 'Distance' is 'seuclidean'. Default is NANSTD(X).
%  
%    'BucketSize'   The maximum number of data points in the leaf node of the
%                   kd-tree (default is 50). This argument is only
%                   meaningful when kd-tree is used for finding nearest
%                   neighbors.
%
%    'SortIndices'  A flag to indicate if output distances and the
%                   corresponding indices should be sorted in the order of 
%                   distances ranging from the smallest to the largest 
%                   distance.  Default is true.
%
%    'CacheSize'    A positive scalar or 'maximal'. The default is 1e3. 
%                   This argument is only meaningful when the alternative
%                   algorithm of computing Euclidean distance is used which
%                   requires an intermediate matrix (when 'NSMethod' is
%                   'exhaustive', and 'Distance' is one of
%                   {'fasteuclidean','fastseuclidean'}).
%                   If numeric, this argument specifies the cache size in 
%                   megabytes (MB) to allocate for an intermediate matrix. 
%                   If 'maximal', knnsearch attempts to allocate enough  
%                   memory for an entire intermediate matrix whose size 
%                   is MX-by-MY (MX is the number of rows of the input 
%                   data X, and MY is the number of rows of the input
%                   data Y).
%                   'CacheSize' does not have to be large enough for an 
%                   entire intermediate matrix, but must be at least large
%                   enough to hold an MX-by-1 vector. Otherwise, the 
%                   regular algorithm of computing Euclidean distance will
%                   be used instead. If the specified cache size exceeds
%                   the available memory, MATLAB issues an out-of-memory
%                   error.
% 
%   Example:
%      % Find 2 nearest neighbors in X and the corresponding values to each
%      % point in Y using the distance metric 'cityblock'
%      X = randn(100,5);
%      Y = randn(25, 5);
%      [idx, dist] = knnsearch(X,Y,'dist','cityblock','k',2);
%
%   See also CREATENS, ExhaustiveSearcher, KDTreeSearcher, RANGESEARCH.

%   Copyright 2009-2023 The MathWorks, Inc.



if nargin > 2
    [varargin{:}] = convertStringsToChars(varargin{:});
end

pnames = { 'k' 'nsmethod' 'bucketsize', 'includeties' 'sortindices' 'cachesize'};
dflts =  { 1   []         []             false         true         1e3};
[numNN, nsmethod, bSize, includeTies,doSort, cacheSize, ~,args] =...
     internal.stats.parseArgs(pnames, dflts, varargin{:});

% NSMethod 'hnsw' should not be used within the knnsearch function.
% Additional checks for the NSMethod argument will be performed during the
% subsequent invocation of the createns function.
if ischar(nsmethod) && strncmpi(nsmethod, 'hnsw', numel(nsmethod))
    error(message('stats:knnsearch:HnswNotSupported'));
end

obj=createns(X,args{:},'nsmethod', nsmethod,'bucketSize',bSize);

if isa(obj,'ExhaustiveSearcher')
    cacheArg = {'CacheSize', cacheSize};
else
    cacheArg = {};
end

if nargout < 2
    idx = knnsearch(obj,Y,'k',numNN, 'includeties', includeTies,...
        'sortindices', doSort, cacheArg{:});
else
    [idx, dist] = knnsearch(obj,Y,'k',numNN, 'includeties',includeTies,...
        'sortindices', doSort, cacheArg{:});
end
end
