%% Full2Full
%% SCAPE
data_dir = 'testdata/scape/shape/';
files = dir(fullfile(data_dir, '*.off'));  % 获取所有 .off 文件
data_name_list = {files.name}';  % 提取文件名
num_of_data = 52:71;
[V,Faces] = load_raw_data(data_dir,data_name_list);
vts_5k = {};
for i = num_of_data
    vts = load(['testdata/scape/corres/mesh', num2str(i,'%03d'), '.vts']);
    vts_5k{1,i} = vts;
end
M = load('testdata/scape/M/M_scape_test.mat').M;
errors1 = [];
arr = [];
for tar = 52:71
    for src = 52:71
        if src == tar
            arr(src-51,tar-51)=0;
            continue
        end
        siz = size(V{src-50});
        siz = siz(2);
        Pidx = ones(siz,1);
        Partial_idx_S = ones(size(Pidx,1),1);
        phiS = load(['testdata/cvpr25/Deformer_Uni3FC_lr_2e-3_bt_2_scape_conv_corr_scape/feature/usefeature_mesh0',num2str(src),'.mat']).uphi; 
        phiT= load(['testdata/cvpr25/Deformer_Uni3FC_lr_2e-3_bt_2_scape_conv_corr_scape/feature/usefeature_mesh0',num2str(tar),'.mat']).uphi;
        basis_full = zeros(size(Partial_idx_S,1),size(phiS,2));
        basis_full(Partial_idx_S > 0,:) = phiS;
        basis_full_vts = basis_full(vts_5k{src},:);
        basis_full_vts_idx = Partial_idx_S(vts_5k{src}); 
        basis_full_vts_partial = basis_full_vts(basis_full_vts_idx>0,:);
        vts_tar = vts_5k{tar};
        gt_idx_T = vts_tar(basis_full_vts_idx>0);
        [idx,distance] = knnsearch(phiT,basis_full_vts_partial); 
        M_T =  M{tar-51};
        ind = sub2ind(size(M_T), idx , gt_idx_T);
        geo_err = M_T(ind);
        geo_err = geo_err(:); 
        errors1  =  [errors1 ; geo_err];
        arr(src-51,tar-51)=mean(geo_err);
    end
end
errors1 = errors1(:);
avg = calculateAverage(arr)
%% FAUST
data_dir = 'testdata/faust/shape/shape_test/';
files = dir(fullfile(data_dir, '*.off'));  % 获取所有 .off 文件
data_name_list = {files.name}';  % 提取文件名
num_of_data = 80:99;
[V,Faces] = load_raw_data(data_dir,data_name_list);
vts_5k = {};
for i = num_of_data
    vts = load(['testdata/faust/corres/tr_reg_', num2str(i,'%03d'), '.vts']);
    vts_5k{1,i} = vts;
end
M = load('testdata/faust/M/M_faust_test.mat').M;
errors2 = [];
arr = [];
for tar = 80:99
    for src = 80:99
        if src == tar
            arr(src-79,tar-79)=0;
            continue
        end
        siz = size(V{src-79});
        siz = siz(2);
        Pidx = ones(siz,1);
        Partial_idx_S = ones(size(Pidx,1),1);
        % phiS = load(['testdata/cvpr25/se-ornet-result/faust-faust/feature/usefeature_',num2str(src),'.mat']).uphi; 
        % phiT= load(['testdata/cvpr25/se-ornet-result/faust-faust/feature/usefeature_',num2str(tar),'.mat']).uphi;
        % phiS = load(['testdata/cvpr25/DPC-result/faust-scape/feature/usefeature_',num2str(src),'.mat']).uphi; 
        % phiT= load(['testdata/cvpr25/DPC-result/faust-scape/feature/usefeature_',num2str(tar),'.mat']).uphi;
        phiS = load(['testdata/cvpr25/Deformer_Uni3FC_lr_2e-3_bt_2_scape_conv_corr_faust/feature/usefeature_tr_reg_0',num2str(src),'.mat']).uphi; 
        phiT= load(['testdata/cvpr25/Deformer_Uni3FC_lr_2e-3_bt_2_scape_conv_corr_faust/feature/usefeature_tr_reg_0',num2str(tar),'.mat']).uphi;
        basis_full = zeros(size(Partial_idx_S,1),size(phiS,2));
        basis_full(Partial_idx_S > 0,:) = phiS;
        basis_full_vts = basis_full(vts_5k{src},:);
        basis_full_vts_idx = Partial_idx_S(vts_5k{src}); 
        basis_full_vts_partial = basis_full_vts(basis_full_vts_idx>0,:);
        vts_tar = vts_5k{tar};
        gt_idx_T = vts_tar(basis_full_vts_idx>0);
        [idx,distance] = knnsearch(phiT,basis_full_vts_partial); 
        M_T =  M{tar-79};
        ind = sub2ind(size(M_T), idx , gt_idx_T);
        geo_err = M_T(ind);
        geo_err = geo_err(:); 
        errors2  =  [errors2 ; geo_err];
        arr(src-79,tar-79)=mean(geo_err);
    end
end
errors2 = errors2(:);
avg = calculateAverage(arr)
%% SHREC19_r
for i = 1:44
    M{i} = load(['testdata/shrec19_r/M/',num2str(i),'.mat']).M;
end
data_dir = 'testdata/shrec19_r/corres/';                  
files = dir(fullfile(data_dir, '*.map'));
data_name_list = {files.name}';
errors3 = []
for num = 1:430
    originalStr = data_name_list{num};
    noSuffixStr = strrep(originalStr, '.map', '')
    splitStr = split(noSuffixStr, '_');
    src = splitStr{1};
    tar = splitStr{2};
    T = load(['testdata/cvpr25/Deformer_Uni3FC_lr_2e-3_bt_2_scape_conv_corr_shrec19_r/T/T_',noSuffixStr,'.txt']);
    Tcorr = load(['testdata/shrec19_r/corres/', noSuffixStr ,'.map']);
    % 以'_'为界分割字符串
    splitStr = split(noSuffixStr, '_');
    num2 = str2num(splitStr{2})
    M_T =  M{num2};
    ind = sub2ind(size(M_T), T , Tcorr);
    geo_err = M_T(ind);
    geo_err = geo_err(:); 
    err(num) = mean(geo_err);
    errors3  =  [errors3; geo_err];
end
errors3 = errors3(:);
mean(errors3)
%% SHREC07
data_dir = 'testdata/shrec07/shape/';
files = dir(fullfile(data_dir, '*.off')); 
data_name_list = {files.name}';
num_of_data = 1:20;
[V,Faces] = load_raw_data(data_dir,data_name_list);
vts_5k = {};
for i = num_of_data
    vts = load(['testdata/shrec07/corres/', num2str(i), '.vts']);
    vts_5k{1,i} = vts;
end
for i = 1:20
    M{i} = load(['testdata/shrec07/M/M',num2str(i),'.mat']).M;
end
errors2 = [];
arr = [];
for tar = 1:20
    for src = 1:20
        if src == tar
            arr(src,tar)=0;
            continue
        end
        siz = size(V{src});
        siz = siz(2);
        Pidx = ones(siz,1);
        Partial_idx_S = ones(size(Pidx,1),1);
        %phiS = load(['testdata/nips/DPC-result/shrec07-scape/feature/usefeature_',num2str(src-1),'.mat']).uphi; 
        %phiT= load(['testdata/nips/DPC-result/shrec07-scape/feature/usefeature_',num2str(tar-1),'.mat']).uphi;
        phiS = load(['testdata/cvpr25/Deformer_Uni3FC_lr_2e-3_bt_2_scape_conv_corr_shrec07/feature/usefeature_',num2str(src),'.mat']).uphi; 
        phiT= load(['testdata/cvpr25/Deformer_Uni3FC_lr_2e-3_bt_2_scape_conv_corr_shrec07/feature/usefeature_',num2str(tar),'.mat']).uphi;
        basis_full = zeros(size(Partial_idx_S,1),size(phiS,2));%5000*128
        basis_full(Partial_idx_S > 0,:) = phiS;
        basis_full_vts = basis_full(vts_5k{src}+1,:);
        basis_full_vts_idx = Partial_idx_S(vts_5k{src}+1); 
        basis_full_vts_partial = basis_full_vts(basis_full_vts_idx>0,:);
        vts_tar = vts_5k{tar}+1;
        gt_idx_T = vts_tar(basis_full_vts_idx>0);
        [idx,distance] = knnsearch(phiT,basis_full_vts_partial); 
        M_T =  M{tar};
        ind = sub2ind(size(M_T), idx , gt_idx_T);
        geo_err = M_T(ind);
        geo_err = geo_err(:); 
        errors2  =  [errors2 ; geo_err];
        arr(src,tar)=mean(geo_err);
    end
end
errors2 = errors2(:);
avg = calculateAverage(arr)
%% Full2Full
%% DT4D
index{1} = 1:3;
index{2} = 4:13;
index{3} = 14:20;
index{4} = 21:23;
index{5} = 24:33;
index{6} = 34:43;
index{7} = 44:46;
index{8} = 47:53;
index{9} = 56:59;
index{10} = 60:62;
index{11} = 63:67;
index{12} = 68:70;
index{13} = 71:79;
M = load('testdata/DT4D/M-Standing2HMagicAttack01034.mat').dist;
for k = 1:13
    num_of_data = index{k};
    data_dir = 'testdata/DT4D/shapes_train/';   
    files = dir(fullfile(data_dir, '*.off')); 
    data_name_list = {files.name}'; 
    [V,Faces] = load_raw_data(data_dir,data_name_list);
    for i = num_of_data
        data_name_list{i} = data_name_list{i}(1:end-4);
    end
    [V1,Faces1] = read_off('testdata/DT4D/Standing2HMagicAttack01034.off');
    vts_5k = {};
    for i = num_of_data
        vts = load(['testdata/DT4D/corres_all/', data_name_list{i}, '.vts']);
        vts_5k{1,i} = vts;
    end
    vts_tar = load('testdata/DT4D/corres_all/Standing2HMagicAttack01034.vts');
    
    errors = [];
    tar = 0;
    for src = num_of_data
        for tar = num_of_data
            siz = size(V{src});
            siz = siz(2);
            Pidx = ones(siz,1);
            Partial_idx_S = ones(size(Pidx,1),1);
            %phiS = load(['testdata/nips/DPC-result/dt4d-faust/feature/usefeature_',num2str(src-1),'.mat']).uphi; 
            %phiT= load(['testdata/nips/DPC-result/dt4d-faust/feature/usefeature_54.mat']).uphi;
            phiS = load(['testdata/cvpr25/Deformer_Uni3FC_lr_2e-3_bt_2_scape_conv_corr_dt4d/feature/usefeature_',data_name_list{src},'.mat']).uphi; 
            phiT= load(['testdata/cvpr25/Deformer_Uni3FC_lr_2e-3_bt_2_scape_conv_corr_dt4d/feature/usefeature_Standing2HMagicAttack01034.mat']).uphi;
            basis_full = zeros(size(Partial_idx_S,1),size(phiS,2));
            basis_full(Partial_idx_S > 0,:) = phiS;
            basis_full_vts = basis_full(vts_5k{src},:);
            basis_full_vts_idx = Partial_idx_S(vts_5k{src}); 
            basis_full_vts_partial = basis_full_vts(basis_full_vts_idx>0,:);
            [idx1,distance] = knnsearch(phiT,basis_full_vts_partial); 

            siz = size(V{tar});
            siz = siz(2);
            Pidx = ones(siz,1);
            Partial_idx_S = ones(size(Pidx,1),1);
            phiS = load(['testdata/cvpr25/Deformer_Uni3FC_lr_2e-3_bt_2_scape_conv_corr_dt4d/feature/usefeature_',data_name_list{tar},'.mat']).uphi; 
            phiT= load(['testdata/cvpr25/Deformer_Uni3FC_lr_2e-3_bt_2_scape_conv_corr_dt4d/feature/usefeature_Standing2HMagicAttack01034.mat']).uphi;
            basis_full = zeros(size(Partial_idx_S,1),size(phiS,2));
            basis_full(Partial_idx_S > 0,:) = phiS;
            basis_full_vts = basis_full(vts_5k{tar},:);
            basis_full_vts_idx = Partial_idx_S(vts_5k{tar}); 
            basis_full_vts_partial = basis_full_vts(basis_full_vts_idx>0,:);
            [idx2,distance] = knnsearch(phiT,basis_full_vts_partial); 
    
            M_T =  M;
            ind = sub2ind(size(M_T), idx1, idx2);
            geo_err = M_T(ind);
            geo_err = geo_err(:); 
            errors  =  [errors ; geo_err];
        end
    end
    errors = errors(:);
err(k) = mean(errors)
end
%% topkids
data_dir = 'testdata/topkids/shapes/';
files = dir(fullfile(data_dir, '*.off'));
data_name_list = {files.name}';
[V,Faces] = load_raw_data(data_dir,data_name_list);
corr = {};
for i = 2:26
    file_name = strrep(data_name_list{i},'.off','');
    data = load(['testdata/topkids/corres/',file_name,'_ref.txt']);
    second_column = data(:, 2);
    corr{1,i} = second_column;
end
M_T = load('testdata/topkids/kid00.mat').dist;
errors1 = [];
arr = [];
for tar = 2:26
    T_gt = corr{tar};
    T_corr = load(['testdata/cvpr25/Deformer_Uni3FC_lr_2e-3_bt_2_scape_conv_corr_topkids/T/T_', strrep(data_name_list{tar},'.off','') ,'_kid00.txt']);
    ind = sub2ind(size(M_T), T_gt , T_corr);
    geo_err = M_T(ind);
    geo_err = geo_err(:); 
    errors1  =  [errors1 ; geo_err];
    arr(tar-1) = mean(geo_err)
end
errors1 = errors1(:);
mean(errors1)





%% Partial2Full
%% SCAPE-Partial-12view
data_dir = 'testdata/scape/shape/';
files = dir(fullfile(data_dir, '*.off'));
data_name_list = {files.name}';
num_of_data = 52:71;
[V,Faces] = load_raw_data(data_dir,data_name_list);
vts_5k = {};
for i = num_of_data
    vts = load(['testdata/scape/corres/mesh', num2str(i,'%03d'), '.vts']);
    vts_5k{1,i} = vts;
end
M = load('testdata/scape/M/M_scape_test.mat').M;
[V1,Faces1] = read_off('testdata/scape/shape/mesh000.off');
vts_tar = load('testdata/scape/corres/mesh000.vts');
M0 = load('testdata/scape/M/M_scape_000.mat').M;
for numview = 1:12
    tar = 0;
    tal = 0;
    num = 0;
    for src = 52:71
        errors1 = [];
        indx = dlmread(['testdata/cvpr25/Deformer_Uni3FC_lr_2e-3_bt_5_scape_conv_partial_epoch20_scape_partial/Deformer_Uni3FC_lr_2e-3_bt_5_scape_conv_partial_epoch20_scape_partial',num2str(numview),'/index_partial/index_mesh0',num2str(src),'.txt'])+1;
        sz = size(indx);
        Pidx = ones(5005,1);
        Partial_idx_S = zeros(size(Pidx,1),1);
        phiS = load(['testdata/cvpr25/Deformer_Uni3FC_lr_2e-3_bt_5_scape_conv_partial_epoch20_scape_partial/Deformer_Uni3FC_lr_2e-3_bt_5_scape_conv_partial_epoch20_scape_partial',num2str(numview),'/feature/usefeature_mesh0',num2str(src),'.mat']);
        phiS = phiS.uphi;
        phiT= load(['testdata/cvpr25/Deformer_Uni3FC_lr_2e-3_bt_5_scape_conv_partial_epoch20_scape_partial/Deformer_Uni3FC_lr_2e-3_bt_5_scape_conv_partial_epoch20_scape_partial',num2str(numview),'/feature/usefeature_mesh000.mat']);
        phiT = phiT.uphi;
        basis_full = zeros(size(Partial_idx_S,1),size(phiS,2));
        basis_full(Partial_idx_S > 0,:) = phiS;
        basis_full_vts = basis_full(vts_5k{src},:);
        basis_full_vts_idx = Partial_idx_S(vts_5k{src}); 
        basis_full_vts_partial = basis_full_vts(basis_full_vts_idx>0,:);
        gt_idx_T = vts_tar(basis_full_vts_idx>0);
        
        [idx,distance] = knnsearch(phiT,basis_full_vts_partial); 
        M_T =  M0;
        ind = sub2ind(size(M_T), idx , gt_idx_T);
        geo_err = M_T(ind);
        geo_err = geo_err(:); 
        errors1  =  [errors1 ; geo_err];
        val{src-51} = mean(errors1);
        tal = tal + mean(errors1)*sz(1);
        num = num + sz(1);
    end
    meanval(numview) = tal/num
end
mean(meanval)
%% FAUST-Partial-12view
% PV的 T + 1
data_dir = 'testdata/faust/shape/shape_test';                       
files = dir(fullfile(data_dir, '*.off'));
data_name_list = {files.name}';
num_of_data = 80:99;
vts_5k = {};
for i = num_of_data
    vts = load(['testdata/faust/corres/tr_reg_', num2str(i,'%03d'), '.vts']);
    vts_5k{1,i} = vts;
end
M0 = load('testdata/faust/M/M_faust_070.mat').M;
[V1,Faces1] = read_off('testdata/faust/shape/tr_reg_070.off');
vts_tar = load('testdata/faust/corres/tr_reg_070.vts');

for numview = 1:12
    tar = 0;
    tal = 0;
    num = 0;
    for src = 80:99
        errors1 = [];
        indx = dlmread(['testdata/cvpr25/Deformer_Uni3FC_lr_2e-3_bt_5_scape_conv_partial_epoch20_faust_partial/Deformer_Uni3FC_lr_2e-3_bt_5_scape_conv_partial_epoch20_faust_partial',num2str(numview),'/index_partial/index_tr_reg_0',num2str(src), '.txt'])+1;
        sz = size(indx);
        Pidx = ones(5005,1);
        Partial_idx_S = zeros(size(Pidx,1),1);
        Partial_idx_S(indx) = 1;
        phiS = load(['testdata/cvpr25/Deformer_Uni3FC_lr_2e-3_bt_5_scape_conv_partial_epoch20_faust_partial/Deformer_Uni3FC_lr_2e-3_bt_5_scape_conv_partial_epoch20_faust_partial',num2str(numview),'/feature/usefeature_tr_reg_0',num2str(src),'.mat']);
        phiS = phiS.uphi;
        %phiT= load(['testdata/nips/se-ornet-result/faust_pv-scape_pv/feature/usefeature_0.mat']);
        phiT= load(['testdata/cvpr25/Deformer_Uni3FC_lr_2e-3_bt_5_scape_conv_partial_epoch20_faust_partial/Deformer_Uni3FC_lr_2e-3_bt_5_scape_conv_partial_epoch20_faust_partial',num2str(numview),'/feature/usefeature_tr_reg_070.mat']);
        phiT = phiT.uphi;
        
        basis_full = zeros(size(Partial_idx_S,1),size(phiS,2));
        basis_full(Partial_idx_S > 0,:) = phiS;
        basis_full_vts = basis_full(vts_5k{src},:);
        basis_full_vts_idx = Partial_idx_S(vts_5k{src});    
        basis_full_vts_partial = basis_full_vts(basis_full_vts_idx>0,:);
        gt_idx_T = vts_tar(basis_full_vts_idx>0);
        
        [idx,distance] = knnsearch(phiT,basis_full_vts_partial); 
        M_T =  M0;
        ind = sub2ind(size(M_T), idx , gt_idx_T);
        geo_err = M_T(ind);
        geo_err = geo_err(:); 
        errors1  =  [errors1 ; geo_err];
        val{src-51} = mean(errors1);
        tal = tal + mean(errors1)*sz(1);
        num = num + sz(1);
    end
    meanval(numview) = tal/num
end
mean(meanval)
%% SHREC19_r-Partial
for i = 1:44
    M{i} = load(['testdata/shrec19_r/M/',num2str(i),'.mat']).M;
end
data_dir = 'testdata/shrec19_r/corres/';                  
files = dir(fullfile(data_dir, '*.map')); 
data_name_list = {files.name}';
exp = 'SE-ORNET_shrec19_partial_train_shrec19_partial'
errors3 = []
for num = 1:430
    originalStr = data_name_list{num};
    noSuffixStr = strrep(originalStr, '.map', '')
    splitStr = split(noSuffixStr, '_');
    src = splitStr{1};
    tar = splitStr{2};
    filePattern = fullfile('testdata/cvpr25/',exp,'/T/', ['T_',src,'_view_*_',tar,'.txt']);
    files = dir(filePattern).name
    T = load(['testdata/cvpr25/',exp,'/T/',files]);
    Tcorr = load(['testdata/shrec19_r/corres/', noSuffixStr ,'.map']);
    pattern = '_(\d+_view_\d+)_';
    tokens = regexp(files, pattern, 'tokens');
    name_partial = tokens{1}{1};
    index = dlmread(['testdata/cvpr25/',exp,'/index/', 'index_',name_partial, '.txt']);
    Tcorr = Tcorr(index+1,:);
    splitStr = split(noSuffixStr, '_');
    num2 = str2num(splitStr{2})
    M_T =  M{num2};
    ind = sub2ind(size(M_T), T , Tcorr);
    geo_err = M_T(ind);
    geo_err = geo_err(:); 
    err(num) = mean(geo_err);
    errors3  =  [errors3; geo_err];
end
errors3 = errors3(:);
mean(errors3)


%% Visualization
S1 = read_off_point('testdata/scape/shape/mesh000.off');
numview = 1
for i = 52:71
    figure(); 
    st = ['testdata/scape/shape/mesh0', num2str(i) ,'.off'];
    indx = dlmread(['testdata/cvpr25/Deformer_Uni3FC_lr_2e-3_bt_5_scape_conv_partial_epoch20_scape_partial/Deformer_Uni3FC_lr_2e-3_bt_5_scape_conv_partial_epoch20_faust_partial',num2str(numview),'/index_partial/index_mesh0',num2str(i), '.txt']);
    S2 = read_off_point(st);
    P2 = S1.surface.VERT;
    P1 = S2.surface.VERT;
    P1 = P1(indx+1,:);
    t = ['testdata/cvpr25/Deformer_Uni3FC_lr_2e-3_bt_5_scape_conv_partial_epoch20_scape_partial/Deformer_Uni3FC_lr_2e-3_bt_5_scape_conv_partial_epoch20_faust_partial',num2str(numview),'/T/T_mesh0', num2str(i) ,'_mesh000.txt'];
    T12_ours = load(t);
    visualize_map_pcd(P1, P2, T12_ours,50*ones(length(P1), 1), [0,0,90],[0, 90]);
end

%% func:cal num
function avg = calculateAverage(arr)
    [n,~] = size(arr);
    % n = 19;
    sum = 0;
    count = 0;
    for i=1:n
        for j=1:n
            if i~=j % 排除对角线元素
                sum = sum + arr(i,j);
                count = count + 1;
            end
        end
    end
    avg = sum / count
end
