img = im2double(imread('~\data\checkerboard.tiff'));

% 保存路径 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
abe_path = '~\data\label\'; % path of aberration patch
ori_path = '~\data\input\'; % path of ideal patch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~,~,~] = mkdir(abe_path);
[~,~,~] = mkdir(ori_path);

% interval between two patches, smaller for a more detail estimation
patch_interval = 50;
% the patch size of the estimated FoV
patch_size = 250;
% pad size of the image, recommend (patch_size-patch_interval)/2 for the 
% border FoV of the image.
pad_size = 100;
% H, W
[H, W, ~] = size(img);
% Center of the image
img_center = [(1 + H)/2, (1 + W)/2];
full_length = sqrt((1 - img_center(1))^2 + (1 - img_center(2))^2);

% because of some FoV is in the border of the image, we need pad the image
img_pad = padarray(img, [pad_size pad_size], 'symmetric', 'both');

% go through all the patches
for h_index = 1 : (H/patch_interval)
    for w_index = 1 : (W/patch_interval)
        % the patch range for degradation transfer, note this is the range
        % on the image after padding.
        h_range = (h_index-1)*patch_interval+1 : ...
            h_index*patch_interval+patch_size-patch_interval;
        w_range = (w_index-1)*patch_interval+1 : ...
            w_index*patch_interval+patch_size-patch_interval;
        
        % patch region
        patch = img_pad(h_range, w_range, :);
        % judging the light and dark value of this patch, three channel is
        % seperated here.
        r_input_ch_hist = imhist(patch(:, :, 1));
        g_input_ch_hist = imhist(patch(:, :, 2));
        b_input_ch_hist = imhist(patch(:, :, 3));
        
        [~, sorted_ch_index] = sort(r_input_ch_hist, 'descend');
        [ch_high_value_r, ch_low_value_r] = judge_blk_wht_value(sorted_ch_index);
        
        [~, sorted_ch_index] = sort(g_input_ch_hist, 'descend');
        [ch_high_value_g, ch_low_value_g] = judge_blk_wht_value(sorted_ch_index);
        
        [~, sorted_ch_index] = sort(b_input_ch_hist, 'descend');
        [ch_high_value_b, ch_low_value_b] = judge_blk_wht_value(sorted_ch_index);
        BW = edge(patch(:, :, 2), 'log');

        % judge the input patch according to the patch and the BW (edge info)        
        input = mod_patch_3_channel(patch, BW, ch_high_value_r, ch_low_value_r, ...
                                               ch_high_value_g, ch_low_value_g, ...
                                               ch_high_value_b, ch_low_value_b);
        % transform the data into 16-bit
        input = im2uint16(input);
        patch = im2uint16(patch);
        
        % save the aberration patch and the ideal patch
        abe_name = strcat(abe_path, 'h_', num2str(h_index*patch_interval, '%04d'), ...
                                    '_w_', num2str(w_index*patch_interval, '%04d'), '.tiff');
        ori_name = strcat(ori_path, 'h_', num2str(h_index*patch_interval, '%04d'), ...
                                    '_w_', num2str(w_index*patch_interval, '%04d'), '.tiff');
        imwrite(patch, abe_name); imwrite(input, ori_name);
        % information logger
        formatSpec = 'field of (%03d, %03d) is OK\n';
        fprintf(formatSpec, h_index, w_index);   
    end
end

function [ch_high_value, ch_low_value] = judge_blk_wht_value(sorted_ch_index)
search_index = 1;                                 
if sorted_ch_index(search_index) > 45
    % 是亮值
    ch_high_value = sorted_ch_index(search_index);
    search_index = search_index + 1;
    while abs(ch_high_value - sorted_ch_index(search_index)) < 45
        search_index = search_index + 1;
    end
    ch_low_value = sorted_ch_index(search_index);
else
    % 是暗值
    ch_low_value = sorted_ch_index(search_index);
    search_index = search_index + 1;
    while abs(ch_low_value - sorted_ch_index(search_index)) < 45
        search_index = search_index + 1;
    end
    ch_high_value = sorted_ch_index(search_index);
end
end   

function input = mod_patch(patch, BW, ch_high_value, ch_low_value)
[H, W] = size(patch);
% 对patch填充一下，为了后面的邻域判断
patch_pad = padarray(patch, [1 1], 'both', 'replicate');
BW_pad = padarray(BW, [1 1], 'both', 'replicate');

% 首先对patch进行一个阈值分割，将亮的区域和暗区域分开
% th = graythresh(patch);
border_pixel_val = patch(BW == 1);
th = sum(border_pixel_val, 'all') / length(border_pixel_val);

% 求得border的像素值最大值和最小值，并对边缘的像素值情况进行分类讨论
bdr_pix_val_max = max(border_pixel_val); bdr_pix_val_min = min(border_pixel_val);
bdr_interval = (bdr_pix_val_max - bdr_pix_val_min) / 8; % 分为8类

input = zeros(H, W); % 初始化input
% 对BWmask进行一个膨胀处理，防止阈值分割的错误，先尝试3邻域的膨胀处理
se = strel('disk', 1, 0);
BW_dilated = imdilate(BW, se);
% figure, imshow(BW);
% 遍历patch，进行处理
for h_index = 1:H
    for w_index = 1:W
        if BW_dilated(h_index, w_index) == 0
            % 若BWmask为0，则为正常区域，判断阈值以后设为亮区域或者暗区域
            if patch(h_index, w_index) < th
                % 暗区域
                input(h_index, w_index) = ch_low_value / 255;
            elseif patch(h_index, w_index) >= th
                % 亮区域
                input(h_index, w_index) = ch_high_value / 255;
            else
                error('像素值或者阈值有问题！');
            end
        elseif (BW_dilated(h_index, w_index) == 1)&&(BW(h_index, w_index) == 0)
            % 在边界的邻域附近，可能会出现阈值分割出问题的情况，
            % 需要判断其所处的邻域是暗区域还是亮区域
            % 判断过程中采用3×3的邻域计算结果，这里还是不能使用阈值，
            % 只能通过亮暗区域对应的方位来判断像素值的大小
            nbr = patch_pad(h_index : h_index+2, w_index : w_index+2);
            bw_nbr = BW_pad(h_index : h_index+2, w_index : w_index+2);
            bw_count = bw_nbr(1, 2) + bw_nbr(2, 1) + bw_nbr(2, 3) + bw_nbr(3, 2);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if bw_count == 1
                % 当3×3领域中的上下左右只有一个点是BWmask为1的点时：
                % 在3×3领域的中心上下左右找到BWmask为1的点
                if bw_nbr(1, 2) == 1
                    % 边缘在点的上方，看下面的像素值和阈值的关系来判断亮暗方向
                    if nbr(3, 2) < th
                        % 暗区域
                        input(h_index, w_index) = ch_low_value / 255;
                    else
                        % 亮区域
                        input(h_index, w_index) = ch_high_value / 255;
                    end
                elseif bw_nbr(2, 1) == 1
                    % 边缘在点的左侧，看右面的像素值和阈值的关系来判断亮暗方向
                    if nbr(2, 3) < th
                        % 暗区域
                        input(h_index, w_index) = ch_low_value / 255;
                    else
                        % 亮区域
                        input(h_index, w_index) = ch_high_value / 255;
                    end
                elseif bw_nbr(3, 2) == 1
                    % 边缘在点的下方，看上面的像素值和阈值的关系来判断亮暗方向
                    if nbr(1, 2) < th
                        % 暗区域
                        input(h_index, w_index) = ch_low_value / 255;
                    else
                        % 亮区域
                        input(h_index, w_index) = ch_high_value / 255;
                    end
                elseif bw_nbr(2, 3) == 1
                    % 边缘在点的右侧，看左面的像素值和阈值的关系来判断亮暗方向
                    if nbr(2, 1) < th
                        % 暗区域
                        input(h_index, w_index) = ch_low_value / 255;
                    else
                        % 亮区域
                        input(h_index, w_index) = ch_high_value / 255;
                    end
                end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            elseif bw_count == 2
                % 当3×3领域中的上下左右有两个点是BWmask为1的点时：
                % 统计BW_nbr的行和和列和
                bw_nbr_sum_col = sum(bw_nbr, 1); bw_nbr_sum_row = sum(bw_nbr, 2);
                [max_col, idx_col] = max(bw_nbr_sum_col); [max_row, idx_row] = max(bw_nbr_sum_row);
                if max_col > max_row
                    % 如果列的和的最大值大于行的和的最大值，那么列是边缘位置，对列的索引的反向像素值进行判断
                    if nbr(2, 4-idx_col) < th
                        % 暗区域
                        input(h_index, w_index) = ch_low_value / 255;
                    else
                        % 亮区域
                        input(h_index, w_index) = ch_high_value / 255;
                    end
                elseif max_col < max_row
                    % 如果列的和的最大值小于行的和的最大值，那么行是边缘位置，对行的索引的反向像素值进行判断
                    if nbr(4-idx_row, 2) < th
                        % 暗区域
                        input(h_index, w_index) = ch_low_value / 255;
                    else
                        % 亮区域
                        input(h_index, w_index) = ch_high_value / 255;
                    end
                elseif max_col == max_row
                    % 如果列的和的最大值等于行的和的最大值，那么边缘位置为两个边缘，对两个的索引的反向像素值进行判断
                    if nbr(4-idx_row, 4-idx_col) < th
                        % 暗区域
                        input(h_index, w_index) = ch_low_value / 255;
                    else
                        % 亮区域
                        input(h_index, w_index) = ch_high_value / 255;
                    end
                end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            elseif bw_count == 3
                % 找到那个为零的点，然后判断
                if bw_nbr(1, 2) == 0
                    if nbr(1, 2) < th
                        % 暗区域
                        input(h_index, w_index) = ch_low_value / 255;
                    else
                        % 亮区域
                        input(h_index, w_index) = ch_high_value / 255;
                    end
                elseif bw_nbr(2, 1) == 0
                    if nbr(2, 1) < th
                        % 暗区域
                        input(h_index, w_index) = ch_low_value / 255;
                    else
                        % 亮区域
                        input(h_index, w_index) = ch_high_value / 255;
                    end
                elseif bw_nbr(2, 3) == 0
                    if nbr(2, 3) < th
                        % 暗区域
                        input(h_index, w_index) = ch_low_value / 255;
                    else
                        % 亮区域
                        input(h_index, w_index) = ch_high_value / 255;
                    end
                elseif bw_nbr(3, 2) == 0
                    if nbr(3, 2) < th
                        % 暗区域
                        input(h_index, w_index) = ch_low_value / 255;
                    else
                        % 亮区域
                        input(h_index, w_index) = ch_high_value / 255;
                    end
                end
            elseif bw_count == 4
                % 对于四邻域小块，直接将其置为暗值
                % 暗区域
                input(h_index, w_index) = ch_low_value / 255;
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
        elseif (BW_dilated(h_index, w_index) == 1)&&(BW(h_index, w_index) == 1)
            % 判断到底是独立一点还是和周围的边缘连续的边界点
            bw_nbr = BW_pad(h_index : h_index+2, w_index : w_index+2);
            bw_nbr_sum = sum(bw_nbr, 'all');
            if bw_nbr_sum <= 1
                % 为独立的一点，根据均值判断其周围是亮区域还是暗区域
                nbr = patch_pad(h_index : h_index+2, w_index : w_index+2);
                nbr_mean = mean(nbr, 'all');
                if nbr_mean < th
                    % 暗区域
                    input(h_index, w_index) = ch_low_value / 255;
                else
                    % 亮区域
                    input(h_index, w_index) = ch_high_value / 255;
                end
            else
                
                
                % 若BWmask为1，则为边界过渡区域。判断其像素值为最大值和最小值之间的哪一个区间范围
                if (bdr_pix_val_min + 0*bdr_interval <= patch(h_index, w_index)) && ...
                        (patch(h_index, w_index) < bdr_pix_val_min + 1*bdr_interval)
                    % 边界1st区间，其值为最暗值和最亮值之间的加权和
                    input(h_index, w_index) = (1/9)*ch_high_value / 255 + (8/9)*ch_low_value / 255;
                elseif (bdr_pix_val_min + 1*bdr_interval <= patch(h_index, w_index)) && ...
                        (patch(h_index, w_index) < bdr_pix_val_min + 2*bdr_interval)
                    % 边界2nd区间，其值为最暗值和最亮值之间的加权和
                    input(h_index, w_index) = (2/9)*ch_high_value / 255 + (7/9)*ch_low_value / 255;
                elseif (bdr_pix_val_min + 2*bdr_interval <= patch(h_index, w_index)) && ...
                        (patch(h_index, w_index) < bdr_pix_val_min + 3*bdr_interval)
                    % 边界3rd区间，其值为最暗值和最亮值之间的加权和
                    input(h_index, w_index) = (3/9)*ch_high_value / 255 + (6/9)*ch_low_value / 255;
                elseif (bdr_pix_val_min + 3*bdr_interval <= patch(h_index, w_index)) && ...
                        (patch(h_index, w_index) < bdr_pix_val_min + 4*bdr_interval)
                    % 边界4th区间，其值为最暗值和最亮值之间的加权和
                    input(h_index, w_index) = (4/9)*ch_high_value / 255 + (5/9)*ch_low_value / 255;
                elseif (bdr_pix_val_min + 4*bdr_interval <= patch(h_index, w_index)) && ...
                        (patch(h_index, w_index) < bdr_pix_val_min + 5*bdr_interval)
                    % 边界5th区间，其值为最暗值和最亮值之间的加权和
                    input(h_index, w_index) = (5/9)*ch_high_value / 255 + (4/9)*ch_low_value / 255;
                elseif (bdr_pix_val_min + 5*bdr_interval <= patch(h_index, w_index)) && ...
                        (patch(h_index, w_index) < bdr_pix_val_min + 6*bdr_interval)
                    % 边界6th区间，其值为最暗值和最亮值之间的加权和
                    input(h_index, w_index) = (6/9)*ch_high_value / 255 + (3/9)*ch_low_value / 255;
                elseif (bdr_pix_val_min + 6*bdr_interval <= patch(h_index, w_index)) && ...
                        (patch(h_index, w_index) < bdr_pix_val_min + 7*bdr_interval)
                    % 边界7th区间，其值为最暗值和最亮值之间的加权和
                    input(h_index, w_index) = (7/9)*ch_high_value / 255 + (2/9)*ch_low_value / 255;
                elseif (bdr_pix_val_min + 7*bdr_interval <= patch(h_index, w_index)) && ...
                        (patch(h_index, w_index) <= bdr_pix_val_min + 8*bdr_interval)
                    % 边界8th区间，其值为最暗值和最亮值之间的加权和
                    input(h_index, w_index) = (8/9)*ch_high_value / 255 + (1/9)*ch_low_value / 255;
                end
            end
        end
    end
end

end

function input = mod_patch_3_channel(patch, BW, ch_high_value_r, ch_low_value_r, ...
                                                ch_high_value_g, ch_low_value_g, ...
                                                ch_high_value_b, ch_low_value_b)
% 将输入图像转成灰度图像，用于判断阈值
patch_gray = rgb2gray(patch);
% 图像大小
[H, W, ch] = size(patch);
% 对patch填充一下，为了后面的邻域判断
patch_pad = padarray(patch_gray, [1 1], 'both', 'replicate');
BW_pad = padarray(BW, [1 1], 'both', 'replicate');

% 首先对patch进行一个阈值分割，将亮的区域和暗区域分开
% th = graythresh(patch);
border_pixel_val = patch_gray(BW == 1);
th = sum(border_pixel_val, 'all') / length(border_pixel_val);

% 求得border的像素值最大值和最小值，并对边缘的像素值情况进行分类讨论
bdr_pix_val_max = max(border_pixel_val); bdr_pix_val_min = min(border_pixel_val);
bdr_interval = (bdr_pix_val_max - bdr_pix_val_min) / 8; % 分为8类

input = zeros(H, W, ch); % 初始化input
% 对BWmask进行一个膨胀处理，防止阈值分割的错误，先尝试3邻域的膨胀处理
se = strel('disk', 1, 0);
BW_dilated = imdilate(BW, se);
% figure, imshow(BW);
% 遍历patch，进行处理
for h_index = 1:H
    for w_index = 1:W
        if BW_dilated(h_index, w_index) == 0
            % 若BWmask为0，则为正常区域，判断阈值以后设为亮区域或者暗区域
            if patch_gray(h_index, w_index) < th
                % 暗区域
                input(h_index, w_index, 1) = ch_low_value_r / 255;
                input(h_index, w_index, 2) = ch_low_value_g / 255;
                input(h_index, w_index, 3) = ch_low_value_b / 255;
            elseif patch_gray(h_index, w_index) >= th
                % 亮区域
                input(h_index, w_index, 1) = ch_high_value_r / 255;
                input(h_index, w_index, 2) = ch_high_value_g / 255;
                input(h_index, w_index, 3) = ch_high_value_b / 255;
            else
                error('像素值或者阈值有问题！');
            end
        elseif (BW_dilated(h_index, w_index) == 1)&&(BW(h_index, w_index) == 0)
            % 在边界的邻域附近，可能会出现阈值分割出问题的情况，
            % 需要判断其所处的邻域是暗区域还是亮区域
            % 判断过程中采用3×3的邻域计算结果，这里还是不能使用阈值，
            % 只能通过亮暗区域对应的方位来判断像素值的大小
            nbr = patch_pad(h_index : h_index+2, w_index : w_index+2);
            bw_nbr = BW_pad(h_index : h_index+2, w_index : w_index+2);
            bw_count = bw_nbr(1, 2) + bw_nbr(2, 1) + bw_nbr(2, 3) + bw_nbr(3, 2);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if bw_count == 1
                % 当3×3领域中的上下左右只有一个点是BWmask为1的点时：
                % 在3×3领域的中心上下左右找到BWmask为1的点
                if bw_nbr(1, 2) == 1
                    % 边缘在点的上方，看下面的像素值和阈值的关系来判断亮暗方向
                    if nbr(3, 2) < th
                        % 暗区域
                        input(h_index, w_index, 1) = ch_low_value_r / 255;
                        input(h_index, w_index, 2) = ch_low_value_g / 255;
                        input(h_index, w_index, 3) = ch_low_value_b / 255;
                    else
                        % 亮区域
                        input(h_index, w_index, 1) = ch_high_value_r / 255;
                        input(h_index, w_index, 2) = ch_high_value_g / 255;
                        input(h_index, w_index, 3) = ch_high_value_b / 255;
                    end
                elseif bw_nbr(2, 1) == 1
                    % 边缘在点的左侧，看右面的像素值和阈值的关系来判断亮暗方向
                    if nbr(2, 3) < th
                        % 暗区域
                        input(h_index, w_index, 1) = ch_low_value_r / 255;
                        input(h_index, w_index, 2) = ch_low_value_g / 255;
                        input(h_index, w_index, 3) = ch_low_value_b / 255;
                    else
                        % 亮区域
                        input(h_index, w_index, 1) = ch_high_value_r / 255;
                        input(h_index, w_index, 2) = ch_high_value_g / 255;
                        input(h_index, w_index, 3) = ch_high_value_b / 255;
                    end
                elseif bw_nbr(3, 2) == 1
                    % 边缘在点的下方，看上面的像素值和阈值的关系来判断亮暗方向
                    if nbr(1, 2) < th
                        % 暗区域
                        input(h_index, w_index, 1) = ch_low_value_r / 255;
                        input(h_index, w_index, 2) = ch_low_value_g / 255;
                        input(h_index, w_index, 3) = ch_low_value_b / 255;
                    else
                        % 亮区域
                        input(h_index, w_index, 1) = ch_high_value_r / 255;
                        input(h_index, w_index, 2) = ch_high_value_g / 255;
                        input(h_index, w_index, 3) = ch_high_value_b / 255;
                    end
                elseif bw_nbr(2, 3) == 1
                    % 边缘在点的右侧，看左面的像素值和阈值的关系来判断亮暗方向
                    if nbr(2, 1) < th
                        % 暗区域
                        input(h_index, w_index, 1) = ch_low_value_r / 255;
                        input(h_index, w_index, 2) = ch_low_value_g / 255;
                        input(h_index, w_index, 3) = ch_low_value_b / 255;
                    else
                        % 亮区域
                        input(h_index, w_index, 1) = ch_high_value_r / 255;
                        input(h_index, w_index, 2) = ch_high_value_g / 255;
                        input(h_index, w_index, 3) = ch_high_value_b / 255;
                    end
                end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            elseif bw_count == 2
                % 当3×3领域中的上下左右有两个点是BWmask为1的点时：
                % 统计BW_nbr的行和和列和
                bw_nbr_sum_col = sum(bw_nbr, 1); bw_nbr_sum_row = sum(bw_nbr, 2);
                [max_col, idx_col] = max(bw_nbr_sum_col); [max_row, idx_row] = max(bw_nbr_sum_row);
                if max_col > max_row
                    % 如果列的和的最大值大于行的和的最大值，那么列是边缘位置，对列的索引的反向像素值进行判断
                    if nbr(2, 4-idx_col) < th
                        % 暗区域
                        input(h_index, w_index, 1) = ch_low_value_r / 255;
                        input(h_index, w_index, 2) = ch_low_value_g / 255;
                        input(h_index, w_index, 3) = ch_low_value_b / 255;
                    else
                        % 亮区域
                        input(h_index, w_index, 1) = ch_high_value_r / 255;
                        input(h_index, w_index, 2) = ch_high_value_g / 255;
                        input(h_index, w_index, 3) = ch_high_value_b / 255;
                    end
                elseif max_col < max_row
                    % 如果列的和的最大值小于行的和的最大值，那么行是边缘位置，对行的索引的反向像素值进行判断
                    if nbr(4-idx_row, 2) < th
                        % 暗区域
                        input(h_index, w_index, 1) = ch_low_value_r / 255;
                        input(h_index, w_index, 2) = ch_low_value_g / 255;
                        input(h_index, w_index, 3) = ch_low_value_b / 255;
                    else
                        % 亮区域
                        input(h_index, w_index, 1) = ch_high_value_r / 255;
                        input(h_index, w_index, 2) = ch_high_value_g / 255;
                        input(h_index, w_index, 3) = ch_high_value_b / 255;
                    end
                elseif max_col == max_row
                    % 如果列的和的最大值等于行的和的最大值，那么边缘位置为两个边缘，对两个的索引的反向像素值进行判断
                    if nbr(4-idx_row, 4-idx_col) < th
                        % 暗区域
                        input(h_index, w_index, 1) = ch_low_value_r / 255;
                        input(h_index, w_index, 2) = ch_low_value_g / 255;
                        input(h_index, w_index, 3) = ch_low_value_b / 255;
                    else
                        % 亮区域
                        input(h_index, w_index, 1) = ch_high_value_r / 255;
                        input(h_index, w_index, 2) = ch_high_value_g / 255;
                        input(h_index, w_index, 3) = ch_high_value_b / 255;
                    end
                end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            elseif bw_count == 3
                % 找到那个为零的点，然后判断
                if bw_nbr(1, 2) == 0
                    if nbr(1, 2) < th
                        % 暗区域
                        input(h_index, w_index, 1) = ch_low_value_r / 255;
                        input(h_index, w_index, 2) = ch_low_value_g / 255;
                        input(h_index, w_index, 3) = ch_low_value_b / 255;
                    else
                        % 亮区域
                        input(h_index, w_index, 1) = ch_high_value_r / 255;
                        input(h_index, w_index, 2) = ch_high_value_g / 255;
                        input(h_index, w_index, 3) = ch_high_value_b / 255;
                    end
                elseif bw_nbr(2, 1) == 0
                    if nbr(2, 1) < th
                        % 暗区域
                        input(h_index, w_index, 1) = ch_low_value_r / 255;
                        input(h_index, w_index, 2) = ch_low_value_g / 255;
                        input(h_index, w_index, 3) = ch_low_value_b / 255;
                    else
                        % 亮区域
                        input(h_index, w_index, 1) = ch_high_value_r / 255;
                        input(h_index, w_index, 2) = ch_high_value_g / 255;
                        input(h_index, w_index, 3) = ch_high_value_b / 255;
                    end
                elseif bw_nbr(2, 3) == 0
                    if nbr(2, 3) < th
                        % 暗区域
                        input(h_index, w_index, 1) = ch_low_value_r / 255;
                        input(h_index, w_index, 2) = ch_low_value_g / 255;
                        input(h_index, w_index, 3) = ch_low_value_b / 255;
                    else
                        % 亮区域
                        input(h_index, w_index, 1) = ch_high_value_r / 255;
                        input(h_index, w_index, 2) = ch_high_value_g / 255;
                        input(h_index, w_index, 3) = ch_high_value_b / 255;
                    end
                elseif bw_nbr(3, 2) == 0
                    if nbr(3, 2) < th
                        % 暗区域
                        input(h_index, w_index, 1) = ch_low_value_r / 255;
                        input(h_index, w_index, 2) = ch_low_value_g / 255;
                        input(h_index, w_index, 3) = ch_low_value_b / 255;
                    else
                        % 亮区域
                        input(h_index, w_index, 1) = ch_high_value_r / 255;
                        input(h_index, w_index, 2) = ch_high_value_g / 255;
                        input(h_index, w_index, 3) = ch_high_value_b / 255;
                    end
                end
            elseif bw_count == 4
                % 对于四邻域小块，直接将其置为暗值
                % 暗区域
                input(h_index, w_index, 1) = ch_low_value_r / 255;
                input(h_index, w_index, 2) = ch_low_value_g / 255;
                input(h_index, w_index, 3) = ch_low_value_b / 255;
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
        elseif (BW_dilated(h_index, w_index) == 1)&&(BW(h_index, w_index) == 1)
            % 判断到底是独立一点还是和周围的边缘连续的边界点
            bw_nbr = BW_pad(h_index : h_index+2, w_index : w_index+2);
            bw_nbr_sum = sum(bw_nbr, 'all');
            if bw_nbr_sum <= 1
                % 为独立的一点，根据均值判断其周围是亮区域还是暗区域
                nbr = patch_pad(h_index : h_index+2, w_index : w_index+2);
                nbr_mean = mean(nbr, 'all');
                if nbr_mean < th
                    % 暗区域
                    input(h_index, w_index, 1) = ch_low_value_r / 255;
                    input(h_index, w_index, 2) = ch_low_value_g / 255;
                    input(h_index, w_index, 3) = ch_low_value_b / 255;
                else
                    % 亮区域
                    input(h_index, w_index, 1) = ch_high_value_r / 255;
                    input(h_index, w_index, 2) = ch_high_value_g / 255;
                    input(h_index, w_index, 3) = ch_high_value_b / 255;
                end
            else
                % 若BWmask为1，则为边界过渡区域。判断其像素值为最大值和最小值之间的哪一个区间范围
                if (bdr_pix_val_min + 0*bdr_interval <= patch_gray(h_index, w_index)) && ...
                        (patch_gray(h_index, w_index) < bdr_pix_val_min + 1*bdr_interval)
                    % 边界1st区间，其值为最暗值和最亮值之间的加权和
                    input(h_index, w_index, 1) = (1/9)*ch_high_value_r / 255 + (8/9)*ch_low_value_r / 255;
                    input(h_index, w_index, 2) = (1/9)*ch_high_value_g / 255 + (8/9)*ch_low_value_g / 255;
                    input(h_index, w_index, 3) = (1/9)*ch_high_value_b / 255 + (8/9)*ch_low_value_b / 255;
                    
                elseif (bdr_pix_val_min + 1*bdr_interval <= patch_gray(h_index, w_index)) && ...
                        (patch_gray(h_index, w_index) < bdr_pix_val_min + 2*bdr_interval)
                    % 边界2nd区间，其值为最暗值和最亮值之间的加权和
                    input(h_index, w_index, 1) = (2/9)*ch_high_value_r / 255 + (7/9)*ch_low_value_r / 255;
                    input(h_index, w_index, 2) = (2/9)*ch_high_value_g / 255 + (7/9)*ch_low_value_g / 255;
                    input(h_index, w_index, 3) = (2/9)*ch_high_value_b / 255 + (7/9)*ch_low_value_b / 255;
                    
                elseif (bdr_pix_val_min + 2*bdr_interval <= patch_gray(h_index, w_index)) && ...
                        (patch_gray(h_index, w_index) < bdr_pix_val_min + 3*bdr_interval)
                    % 边界3rd区间，其值为最暗值和最亮值之间的加权和
                    input(h_index, w_index, 1) = (3/9)*ch_high_value_r / 255 + (6/9)*ch_low_value_r / 255;
                    input(h_index, w_index, 2) = (3/9)*ch_high_value_g / 255 + (6/9)*ch_low_value_g / 255;
                    input(h_index, w_index, 3) = (3/9)*ch_high_value_b / 255 + (6/9)*ch_low_value_b / 255;
                    
                elseif (bdr_pix_val_min + 3*bdr_interval <= patch_gray(h_index, w_index)) && ...
                        (patch_gray(h_index, w_index) < bdr_pix_val_min + 4*bdr_interval)
                    % 边界4th区间，其值为最暗值和最亮值之间的加权和
                    input(h_index, w_index, 1) = (4/9)*ch_high_value_r / 255 + (5/9)*ch_low_value_r / 255;
                    input(h_index, w_index, 2) = (4/9)*ch_high_value_g / 255 + (5/9)*ch_low_value_g / 255;
                    input(h_index, w_index, 3) = (4/9)*ch_high_value_b / 255 + (5/9)*ch_low_value_b / 255;
                    
                elseif (bdr_pix_val_min + 4*bdr_interval <= patch_gray(h_index, w_index)) && ...
                        (patch_gray(h_index, w_index) < bdr_pix_val_min + 5*bdr_interval)
                    % 边界5th区间，其值为最暗值和最亮值之间的加权和
                    input(h_index, w_index, 1) = (5/9)*ch_high_value_r / 255 + (4/9)*ch_low_value_r / 255;
                    input(h_index, w_index, 2) = (5/9)*ch_high_value_g / 255 + (4/9)*ch_low_value_g / 255;
                    input(h_index, w_index, 3) = (5/9)*ch_high_value_b / 255 + (4/9)*ch_low_value_b / 255;
                    
                elseif (bdr_pix_val_min + 5*bdr_interval <= patch_gray(h_index, w_index)) && ...
                        (patch_gray(h_index, w_index) < bdr_pix_val_min + 6*bdr_interval)
                    % 边界6th区间，其值为最暗值和最亮值之间的加权和
                    input(h_index, w_index, 1) = (6/9)*ch_high_value_r / 255 + (3/9)*ch_low_value_r / 255;
                    input(h_index, w_index, 2) = (6/9)*ch_high_value_g / 255 + (3/9)*ch_low_value_g / 255;
                    input(h_index, w_index, 3) = (6/9)*ch_high_value_b / 255 + (3/9)*ch_low_value_b / 255;
                    
                elseif (bdr_pix_val_min + 6*bdr_interval <= patch_gray(h_index, w_index)) && ...
                        (patch_gray(h_index, w_index) < bdr_pix_val_min + 7*bdr_interval)
                    % 边界7th区间，其值为最暗值和最亮值之间的加权和
                    input(h_index, w_index, 1) = (7/9)*ch_high_value_r / 255 + (2/9)*ch_low_value_r / 255;
                    input(h_index, w_index, 2) = (7/9)*ch_high_value_g / 255 + (2/9)*ch_low_value_g / 255;
                    input(h_index, w_index, 3) = (7/9)*ch_high_value_b / 255 + (2/9)*ch_low_value_b / 255;
                    
                elseif (bdr_pix_val_min + 7*bdr_interval <= patch_gray(h_index, w_index)) && ...
                        (patch_gray(h_index, w_index) <= bdr_pix_val_min + 8*bdr_interval)
                    % 边界8th区间，其值为最暗值和最亮值之间的加权和
                    input(h_index, w_index, 1) = (8/9)*ch_high_value_r / 255 + (1/9)*ch_low_value_r / 255;
                    input(h_index, w_index, 2) = (8/9)*ch_high_value_g / 255 + (1/9)*ch_low_value_g / 255;
                    input(h_index, w_index, 3) = (8/9)*ch_high_value_b / 255 + (1/9)*ch_low_value_b / 255;
                    
                end       
            end
        end
    end
end
end