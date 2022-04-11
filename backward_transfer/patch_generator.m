img = im2double(imread('~\data\checkerboard.tiff'));

% ����·�� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    % ����ֵ
    ch_high_value = sorted_ch_index(search_index);
    search_index = search_index + 1;
    while abs(ch_high_value - sorted_ch_index(search_index)) < 45
        search_index = search_index + 1;
    end
    ch_low_value = sorted_ch_index(search_index);
else
    % �ǰ�ֵ
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
% ��patch���һ�£�Ϊ�˺���������ж�
patch_pad = padarray(patch, [1 1], 'both', 'replicate');
BW_pad = padarray(BW, [1 1], 'both', 'replicate');

% ���ȶ�patch����һ����ֵ�ָ����������Ͱ�����ֿ�
% th = graythresh(patch);
border_pixel_val = patch(BW == 1);
th = sum(border_pixel_val, 'all') / length(border_pixel_val);

% ���border������ֵ���ֵ����Сֵ�����Ա�Ե������ֵ������з�������
bdr_pix_val_max = max(border_pixel_val); bdr_pix_val_min = min(border_pixel_val);
bdr_interval = (bdr_pix_val_max - bdr_pix_val_min) / 8; % ��Ϊ8��

input = zeros(H, W); % ��ʼ��input
% ��BWmask����һ�����ʹ�����ֹ��ֵ�ָ�Ĵ����ȳ���3��������ʹ���
se = strel('disk', 1, 0);
BW_dilated = imdilate(BW, se);
% figure, imshow(BW);
% ����patch�����д���
for h_index = 1:H
    for w_index = 1:W
        if BW_dilated(h_index, w_index) == 0
            % ��BWmaskΪ0����Ϊ���������ж���ֵ�Ժ���Ϊ��������߰�����
            if patch(h_index, w_index) < th
                % ������
                input(h_index, w_index) = ch_low_value / 255;
            elseif patch(h_index, w_index) >= th
                % ������
                input(h_index, w_index) = ch_high_value / 255;
            else
                error('����ֵ������ֵ�����⣡');
            end
        elseif (BW_dilated(h_index, w_index) == 1)&&(BW(h_index, w_index) == 0)
            % �ڱ߽�����򸽽������ܻ������ֵ�ָ������������
            % ��Ҫ�ж��������������ǰ�������������
            % �жϹ����в���3��3����������������ﻹ�ǲ���ʹ����ֵ��
            % ֻ��ͨ�����������Ӧ�ķ�λ���ж�����ֵ�Ĵ�С
            nbr = patch_pad(h_index : h_index+2, w_index : w_index+2);
            bw_nbr = BW_pad(h_index : h_index+2, w_index : w_index+2);
            bw_count = bw_nbr(1, 2) + bw_nbr(2, 1) + bw_nbr(2, 3) + bw_nbr(3, 2);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if bw_count == 1
                % ��3��3�����е���������ֻ��һ������BWmaskΪ1�ĵ�ʱ��
                % ��3��3������������������ҵ�BWmaskΪ1�ĵ�
                if bw_nbr(1, 2) == 1
                    % ��Ե�ڵ���Ϸ��������������ֵ����ֵ�Ĺ�ϵ���ж���������
                    if nbr(3, 2) < th
                        % ������
                        input(h_index, w_index) = ch_low_value / 255;
                    else
                        % ������
                        input(h_index, w_index) = ch_high_value / 255;
                    end
                elseif bw_nbr(2, 1) == 1
                    % ��Ե�ڵ����࣬�����������ֵ����ֵ�Ĺ�ϵ���ж���������
                    if nbr(2, 3) < th
                        % ������
                        input(h_index, w_index) = ch_low_value / 255;
                    else
                        % ������
                        input(h_index, w_index) = ch_high_value / 255;
                    end
                elseif bw_nbr(3, 2) == 1
                    % ��Ե�ڵ���·��������������ֵ����ֵ�Ĺ�ϵ���ж���������
                    if nbr(1, 2) < th
                        % ������
                        input(h_index, w_index) = ch_low_value / 255;
                    else
                        % ������
                        input(h_index, w_index) = ch_high_value / 255;
                    end
                elseif bw_nbr(2, 3) == 1
                    % ��Ե�ڵ���Ҳ࣬�����������ֵ����ֵ�Ĺ�ϵ���ж���������
                    if nbr(2, 1) < th
                        % ������
                        input(h_index, w_index) = ch_low_value / 255;
                    else
                        % ������
                        input(h_index, w_index) = ch_high_value / 255;
                    end
                end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            elseif bw_count == 2
                % ��3��3�����е�������������������BWmaskΪ1�ĵ�ʱ��
                % ͳ��BW_nbr���кͺ��к�
                bw_nbr_sum_col = sum(bw_nbr, 1); bw_nbr_sum_row = sum(bw_nbr, 2);
                [max_col, idx_col] = max(bw_nbr_sum_col); [max_row, idx_row] = max(bw_nbr_sum_row);
                if max_col > max_row
                    % ����еĺ͵����ֵ�����еĺ͵����ֵ����ô���Ǳ�Եλ�ã����е������ķ�������ֵ�����ж�
                    if nbr(2, 4-idx_col) < th
                        % ������
                        input(h_index, w_index) = ch_low_value / 255;
                    else
                        % ������
                        input(h_index, w_index) = ch_high_value / 255;
                    end
                elseif max_col < max_row
                    % ����еĺ͵����ֵС���еĺ͵����ֵ����ô���Ǳ�Եλ�ã����е������ķ�������ֵ�����ж�
                    if nbr(4-idx_row, 2) < th
                        % ������
                        input(h_index, w_index) = ch_low_value / 255;
                    else
                        % ������
                        input(h_index, w_index) = ch_high_value / 255;
                    end
                elseif max_col == max_row
                    % ����еĺ͵����ֵ�����еĺ͵����ֵ����ô��Եλ��Ϊ������Ե���������������ķ�������ֵ�����ж�
                    if nbr(4-idx_row, 4-idx_col) < th
                        % ������
                        input(h_index, w_index) = ch_low_value / 255;
                    else
                        % ������
                        input(h_index, w_index) = ch_high_value / 255;
                    end
                end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            elseif bw_count == 3
                % �ҵ��Ǹ�Ϊ��ĵ㣬Ȼ���ж�
                if bw_nbr(1, 2) == 0
                    if nbr(1, 2) < th
                        % ������
                        input(h_index, w_index) = ch_low_value / 255;
                    else
                        % ������
                        input(h_index, w_index) = ch_high_value / 255;
                    end
                elseif bw_nbr(2, 1) == 0
                    if nbr(2, 1) < th
                        % ������
                        input(h_index, w_index) = ch_low_value / 255;
                    else
                        % ������
                        input(h_index, w_index) = ch_high_value / 255;
                    end
                elseif bw_nbr(2, 3) == 0
                    if nbr(2, 3) < th
                        % ������
                        input(h_index, w_index) = ch_low_value / 255;
                    else
                        % ������
                        input(h_index, w_index) = ch_high_value / 255;
                    end
                elseif bw_nbr(3, 2) == 0
                    if nbr(3, 2) < th
                        % ������
                        input(h_index, w_index) = ch_low_value / 255;
                    else
                        % ������
                        input(h_index, w_index) = ch_high_value / 255;
                    end
                end
            elseif bw_count == 4
                % ����������С�飬ֱ�ӽ�����Ϊ��ֵ
                % ������
                input(h_index, w_index) = ch_low_value / 255;
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
        elseif (BW_dilated(h_index, w_index) == 1)&&(BW(h_index, w_index) == 1)
            % �жϵ����Ƕ���һ�㻹�Ǻ���Χ�ı�Ե�����ı߽��
            bw_nbr = BW_pad(h_index : h_index+2, w_index : w_index+2);
            bw_nbr_sum = sum(bw_nbr, 'all');
            if bw_nbr_sum <= 1
                % Ϊ������һ�㣬���ݾ�ֵ�ж�����Χ���������ǰ�����
                nbr = patch_pad(h_index : h_index+2, w_index : w_index+2);
                nbr_mean = mean(nbr, 'all');
                if nbr_mean < th
                    % ������
                    input(h_index, w_index) = ch_low_value / 255;
                else
                    % ������
                    input(h_index, w_index) = ch_high_value / 255;
                end
            else
                
                
                % ��BWmaskΪ1����Ϊ�߽���������ж�������ֵΪ���ֵ����Сֵ֮�����һ�����䷶Χ
                if (bdr_pix_val_min + 0*bdr_interval <= patch(h_index, w_index)) && ...
                        (patch(h_index, w_index) < bdr_pix_val_min + 1*bdr_interval)
                    % �߽�1st���䣬��ֵΪ�ֵ������ֵ֮��ļ�Ȩ��
                    input(h_index, w_index) = (1/9)*ch_high_value / 255 + (8/9)*ch_low_value / 255;
                elseif (bdr_pix_val_min + 1*bdr_interval <= patch(h_index, w_index)) && ...
                        (patch(h_index, w_index) < bdr_pix_val_min + 2*bdr_interval)
                    % �߽�2nd���䣬��ֵΪ�ֵ������ֵ֮��ļ�Ȩ��
                    input(h_index, w_index) = (2/9)*ch_high_value / 255 + (7/9)*ch_low_value / 255;
                elseif (bdr_pix_val_min + 2*bdr_interval <= patch(h_index, w_index)) && ...
                        (patch(h_index, w_index) < bdr_pix_val_min + 3*bdr_interval)
                    % �߽�3rd���䣬��ֵΪ�ֵ������ֵ֮��ļ�Ȩ��
                    input(h_index, w_index) = (3/9)*ch_high_value / 255 + (6/9)*ch_low_value / 255;
                elseif (bdr_pix_val_min + 3*bdr_interval <= patch(h_index, w_index)) && ...
                        (patch(h_index, w_index) < bdr_pix_val_min + 4*bdr_interval)
                    % �߽�4th���䣬��ֵΪ�ֵ������ֵ֮��ļ�Ȩ��
                    input(h_index, w_index) = (4/9)*ch_high_value / 255 + (5/9)*ch_low_value / 255;
                elseif (bdr_pix_val_min + 4*bdr_interval <= patch(h_index, w_index)) && ...
                        (patch(h_index, w_index) < bdr_pix_val_min + 5*bdr_interval)
                    % �߽�5th���䣬��ֵΪ�ֵ������ֵ֮��ļ�Ȩ��
                    input(h_index, w_index) = (5/9)*ch_high_value / 255 + (4/9)*ch_low_value / 255;
                elseif (bdr_pix_val_min + 5*bdr_interval <= patch(h_index, w_index)) && ...
                        (patch(h_index, w_index) < bdr_pix_val_min + 6*bdr_interval)
                    % �߽�6th���䣬��ֵΪ�ֵ������ֵ֮��ļ�Ȩ��
                    input(h_index, w_index) = (6/9)*ch_high_value / 255 + (3/9)*ch_low_value / 255;
                elseif (bdr_pix_val_min + 6*bdr_interval <= patch(h_index, w_index)) && ...
                        (patch(h_index, w_index) < bdr_pix_val_min + 7*bdr_interval)
                    % �߽�7th���䣬��ֵΪ�ֵ������ֵ֮��ļ�Ȩ��
                    input(h_index, w_index) = (7/9)*ch_high_value / 255 + (2/9)*ch_low_value / 255;
                elseif (bdr_pix_val_min + 7*bdr_interval <= patch(h_index, w_index)) && ...
                        (patch(h_index, w_index) <= bdr_pix_val_min + 8*bdr_interval)
                    % �߽�8th���䣬��ֵΪ�ֵ������ֵ֮��ļ�Ȩ��
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
% ������ͼ��ת�ɻҶ�ͼ�������ж���ֵ
patch_gray = rgb2gray(patch);
% ͼ���С
[H, W, ch] = size(patch);
% ��patch���һ�£�Ϊ�˺���������ж�
patch_pad = padarray(patch_gray, [1 1], 'both', 'replicate');
BW_pad = padarray(BW, [1 1], 'both', 'replicate');

% ���ȶ�patch����һ����ֵ�ָ����������Ͱ�����ֿ�
% th = graythresh(patch);
border_pixel_val = patch_gray(BW == 1);
th = sum(border_pixel_val, 'all') / length(border_pixel_val);

% ���border������ֵ���ֵ����Сֵ�����Ա�Ե������ֵ������з�������
bdr_pix_val_max = max(border_pixel_val); bdr_pix_val_min = min(border_pixel_val);
bdr_interval = (bdr_pix_val_max - bdr_pix_val_min) / 8; % ��Ϊ8��

input = zeros(H, W, ch); % ��ʼ��input
% ��BWmask����һ�����ʹ�����ֹ��ֵ�ָ�Ĵ����ȳ���3��������ʹ���
se = strel('disk', 1, 0);
BW_dilated = imdilate(BW, se);
% figure, imshow(BW);
% ����patch�����д���
for h_index = 1:H
    for w_index = 1:W
        if BW_dilated(h_index, w_index) == 0
            % ��BWmaskΪ0����Ϊ���������ж���ֵ�Ժ���Ϊ��������߰�����
            if patch_gray(h_index, w_index) < th
                % ������
                input(h_index, w_index, 1) = ch_low_value_r / 255;
                input(h_index, w_index, 2) = ch_low_value_g / 255;
                input(h_index, w_index, 3) = ch_low_value_b / 255;
            elseif patch_gray(h_index, w_index) >= th
                % ������
                input(h_index, w_index, 1) = ch_high_value_r / 255;
                input(h_index, w_index, 2) = ch_high_value_g / 255;
                input(h_index, w_index, 3) = ch_high_value_b / 255;
            else
                error('����ֵ������ֵ�����⣡');
            end
        elseif (BW_dilated(h_index, w_index) == 1)&&(BW(h_index, w_index) == 0)
            % �ڱ߽�����򸽽������ܻ������ֵ�ָ������������
            % ��Ҫ�ж��������������ǰ�������������
            % �жϹ����в���3��3����������������ﻹ�ǲ���ʹ����ֵ��
            % ֻ��ͨ�����������Ӧ�ķ�λ���ж�����ֵ�Ĵ�С
            nbr = patch_pad(h_index : h_index+2, w_index : w_index+2);
            bw_nbr = BW_pad(h_index : h_index+2, w_index : w_index+2);
            bw_count = bw_nbr(1, 2) + bw_nbr(2, 1) + bw_nbr(2, 3) + bw_nbr(3, 2);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if bw_count == 1
                % ��3��3�����е���������ֻ��һ������BWmaskΪ1�ĵ�ʱ��
                % ��3��3������������������ҵ�BWmaskΪ1�ĵ�
                if bw_nbr(1, 2) == 1
                    % ��Ե�ڵ���Ϸ��������������ֵ����ֵ�Ĺ�ϵ���ж���������
                    if nbr(3, 2) < th
                        % ������
                        input(h_index, w_index, 1) = ch_low_value_r / 255;
                        input(h_index, w_index, 2) = ch_low_value_g / 255;
                        input(h_index, w_index, 3) = ch_low_value_b / 255;
                    else
                        % ������
                        input(h_index, w_index, 1) = ch_high_value_r / 255;
                        input(h_index, w_index, 2) = ch_high_value_g / 255;
                        input(h_index, w_index, 3) = ch_high_value_b / 255;
                    end
                elseif bw_nbr(2, 1) == 1
                    % ��Ե�ڵ����࣬�����������ֵ����ֵ�Ĺ�ϵ���ж���������
                    if nbr(2, 3) < th
                        % ������
                        input(h_index, w_index, 1) = ch_low_value_r / 255;
                        input(h_index, w_index, 2) = ch_low_value_g / 255;
                        input(h_index, w_index, 3) = ch_low_value_b / 255;
                    else
                        % ������
                        input(h_index, w_index, 1) = ch_high_value_r / 255;
                        input(h_index, w_index, 2) = ch_high_value_g / 255;
                        input(h_index, w_index, 3) = ch_high_value_b / 255;
                    end
                elseif bw_nbr(3, 2) == 1
                    % ��Ե�ڵ���·��������������ֵ����ֵ�Ĺ�ϵ���ж���������
                    if nbr(1, 2) < th
                        % ������
                        input(h_index, w_index, 1) = ch_low_value_r / 255;
                        input(h_index, w_index, 2) = ch_low_value_g / 255;
                        input(h_index, w_index, 3) = ch_low_value_b / 255;
                    else
                        % ������
                        input(h_index, w_index, 1) = ch_high_value_r / 255;
                        input(h_index, w_index, 2) = ch_high_value_g / 255;
                        input(h_index, w_index, 3) = ch_high_value_b / 255;
                    end
                elseif bw_nbr(2, 3) == 1
                    % ��Ե�ڵ���Ҳ࣬�����������ֵ����ֵ�Ĺ�ϵ���ж���������
                    if nbr(2, 1) < th
                        % ������
                        input(h_index, w_index, 1) = ch_low_value_r / 255;
                        input(h_index, w_index, 2) = ch_low_value_g / 255;
                        input(h_index, w_index, 3) = ch_low_value_b / 255;
                    else
                        % ������
                        input(h_index, w_index, 1) = ch_high_value_r / 255;
                        input(h_index, w_index, 2) = ch_high_value_g / 255;
                        input(h_index, w_index, 3) = ch_high_value_b / 255;
                    end
                end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            elseif bw_count == 2
                % ��3��3�����е�������������������BWmaskΪ1�ĵ�ʱ��
                % ͳ��BW_nbr���кͺ��к�
                bw_nbr_sum_col = sum(bw_nbr, 1); bw_nbr_sum_row = sum(bw_nbr, 2);
                [max_col, idx_col] = max(bw_nbr_sum_col); [max_row, idx_row] = max(bw_nbr_sum_row);
                if max_col > max_row
                    % ����еĺ͵����ֵ�����еĺ͵����ֵ����ô���Ǳ�Եλ�ã����е������ķ�������ֵ�����ж�
                    if nbr(2, 4-idx_col) < th
                        % ������
                        input(h_index, w_index, 1) = ch_low_value_r / 255;
                        input(h_index, w_index, 2) = ch_low_value_g / 255;
                        input(h_index, w_index, 3) = ch_low_value_b / 255;
                    else
                        % ������
                        input(h_index, w_index, 1) = ch_high_value_r / 255;
                        input(h_index, w_index, 2) = ch_high_value_g / 255;
                        input(h_index, w_index, 3) = ch_high_value_b / 255;
                    end
                elseif max_col < max_row
                    % ����еĺ͵����ֵС���еĺ͵����ֵ����ô���Ǳ�Եλ�ã����е������ķ�������ֵ�����ж�
                    if nbr(4-idx_row, 2) < th
                        % ������
                        input(h_index, w_index, 1) = ch_low_value_r / 255;
                        input(h_index, w_index, 2) = ch_low_value_g / 255;
                        input(h_index, w_index, 3) = ch_low_value_b / 255;
                    else
                        % ������
                        input(h_index, w_index, 1) = ch_high_value_r / 255;
                        input(h_index, w_index, 2) = ch_high_value_g / 255;
                        input(h_index, w_index, 3) = ch_high_value_b / 255;
                    end
                elseif max_col == max_row
                    % ����еĺ͵����ֵ�����еĺ͵����ֵ����ô��Եλ��Ϊ������Ե���������������ķ�������ֵ�����ж�
                    if nbr(4-idx_row, 4-idx_col) < th
                        % ������
                        input(h_index, w_index, 1) = ch_low_value_r / 255;
                        input(h_index, w_index, 2) = ch_low_value_g / 255;
                        input(h_index, w_index, 3) = ch_low_value_b / 255;
                    else
                        % ������
                        input(h_index, w_index, 1) = ch_high_value_r / 255;
                        input(h_index, w_index, 2) = ch_high_value_g / 255;
                        input(h_index, w_index, 3) = ch_high_value_b / 255;
                    end
                end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            elseif bw_count == 3
                % �ҵ��Ǹ�Ϊ��ĵ㣬Ȼ���ж�
                if bw_nbr(1, 2) == 0
                    if nbr(1, 2) < th
                        % ������
                        input(h_index, w_index, 1) = ch_low_value_r / 255;
                        input(h_index, w_index, 2) = ch_low_value_g / 255;
                        input(h_index, w_index, 3) = ch_low_value_b / 255;
                    else
                        % ������
                        input(h_index, w_index, 1) = ch_high_value_r / 255;
                        input(h_index, w_index, 2) = ch_high_value_g / 255;
                        input(h_index, w_index, 3) = ch_high_value_b / 255;
                    end
                elseif bw_nbr(2, 1) == 0
                    if nbr(2, 1) < th
                        % ������
                        input(h_index, w_index, 1) = ch_low_value_r / 255;
                        input(h_index, w_index, 2) = ch_low_value_g / 255;
                        input(h_index, w_index, 3) = ch_low_value_b / 255;
                    else
                        % ������
                        input(h_index, w_index, 1) = ch_high_value_r / 255;
                        input(h_index, w_index, 2) = ch_high_value_g / 255;
                        input(h_index, w_index, 3) = ch_high_value_b / 255;
                    end
                elseif bw_nbr(2, 3) == 0
                    if nbr(2, 3) < th
                        % ������
                        input(h_index, w_index, 1) = ch_low_value_r / 255;
                        input(h_index, w_index, 2) = ch_low_value_g / 255;
                        input(h_index, w_index, 3) = ch_low_value_b / 255;
                    else
                        % ������
                        input(h_index, w_index, 1) = ch_high_value_r / 255;
                        input(h_index, w_index, 2) = ch_high_value_g / 255;
                        input(h_index, w_index, 3) = ch_high_value_b / 255;
                    end
                elseif bw_nbr(3, 2) == 0
                    if nbr(3, 2) < th
                        % ������
                        input(h_index, w_index, 1) = ch_low_value_r / 255;
                        input(h_index, w_index, 2) = ch_low_value_g / 255;
                        input(h_index, w_index, 3) = ch_low_value_b / 255;
                    else
                        % ������
                        input(h_index, w_index, 1) = ch_high_value_r / 255;
                        input(h_index, w_index, 2) = ch_high_value_g / 255;
                        input(h_index, w_index, 3) = ch_high_value_b / 255;
                    end
                end
            elseif bw_count == 4
                % ����������С�飬ֱ�ӽ�����Ϊ��ֵ
                % ������
                input(h_index, w_index, 1) = ch_low_value_r / 255;
                input(h_index, w_index, 2) = ch_low_value_g / 255;
                input(h_index, w_index, 3) = ch_low_value_b / 255;
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
        elseif (BW_dilated(h_index, w_index) == 1)&&(BW(h_index, w_index) == 1)
            % �жϵ����Ƕ���һ�㻹�Ǻ���Χ�ı�Ե�����ı߽��
            bw_nbr = BW_pad(h_index : h_index+2, w_index : w_index+2);
            bw_nbr_sum = sum(bw_nbr, 'all');
            if bw_nbr_sum <= 1
                % Ϊ������һ�㣬���ݾ�ֵ�ж�����Χ���������ǰ�����
                nbr = patch_pad(h_index : h_index+2, w_index : w_index+2);
                nbr_mean = mean(nbr, 'all');
                if nbr_mean < th
                    % ������
                    input(h_index, w_index, 1) = ch_low_value_r / 255;
                    input(h_index, w_index, 2) = ch_low_value_g / 255;
                    input(h_index, w_index, 3) = ch_low_value_b / 255;
                else
                    % ������
                    input(h_index, w_index, 1) = ch_high_value_r / 255;
                    input(h_index, w_index, 2) = ch_high_value_g / 255;
                    input(h_index, w_index, 3) = ch_high_value_b / 255;
                end
            else
                % ��BWmaskΪ1����Ϊ�߽���������ж�������ֵΪ���ֵ����Сֵ֮�����һ�����䷶Χ
                if (bdr_pix_val_min + 0*bdr_interval <= patch_gray(h_index, w_index)) && ...
                        (patch_gray(h_index, w_index) < bdr_pix_val_min + 1*bdr_interval)
                    % �߽�1st���䣬��ֵΪ�ֵ������ֵ֮��ļ�Ȩ��
                    input(h_index, w_index, 1) = (1/9)*ch_high_value_r / 255 + (8/9)*ch_low_value_r / 255;
                    input(h_index, w_index, 2) = (1/9)*ch_high_value_g / 255 + (8/9)*ch_low_value_g / 255;
                    input(h_index, w_index, 3) = (1/9)*ch_high_value_b / 255 + (8/9)*ch_low_value_b / 255;
                    
                elseif (bdr_pix_val_min + 1*bdr_interval <= patch_gray(h_index, w_index)) && ...
                        (patch_gray(h_index, w_index) < bdr_pix_val_min + 2*bdr_interval)
                    % �߽�2nd���䣬��ֵΪ�ֵ������ֵ֮��ļ�Ȩ��
                    input(h_index, w_index, 1) = (2/9)*ch_high_value_r / 255 + (7/9)*ch_low_value_r / 255;
                    input(h_index, w_index, 2) = (2/9)*ch_high_value_g / 255 + (7/9)*ch_low_value_g / 255;
                    input(h_index, w_index, 3) = (2/9)*ch_high_value_b / 255 + (7/9)*ch_low_value_b / 255;
                    
                elseif (bdr_pix_val_min + 2*bdr_interval <= patch_gray(h_index, w_index)) && ...
                        (patch_gray(h_index, w_index) < bdr_pix_val_min + 3*bdr_interval)
                    % �߽�3rd���䣬��ֵΪ�ֵ������ֵ֮��ļ�Ȩ��
                    input(h_index, w_index, 1) = (3/9)*ch_high_value_r / 255 + (6/9)*ch_low_value_r / 255;
                    input(h_index, w_index, 2) = (3/9)*ch_high_value_g / 255 + (6/9)*ch_low_value_g / 255;
                    input(h_index, w_index, 3) = (3/9)*ch_high_value_b / 255 + (6/9)*ch_low_value_b / 255;
                    
                elseif (bdr_pix_val_min + 3*bdr_interval <= patch_gray(h_index, w_index)) && ...
                        (patch_gray(h_index, w_index) < bdr_pix_val_min + 4*bdr_interval)
                    % �߽�4th���䣬��ֵΪ�ֵ������ֵ֮��ļ�Ȩ��
                    input(h_index, w_index, 1) = (4/9)*ch_high_value_r / 255 + (5/9)*ch_low_value_r / 255;
                    input(h_index, w_index, 2) = (4/9)*ch_high_value_g / 255 + (5/9)*ch_low_value_g / 255;
                    input(h_index, w_index, 3) = (4/9)*ch_high_value_b / 255 + (5/9)*ch_low_value_b / 255;
                    
                elseif (bdr_pix_val_min + 4*bdr_interval <= patch_gray(h_index, w_index)) && ...
                        (patch_gray(h_index, w_index) < bdr_pix_val_min + 5*bdr_interval)
                    % �߽�5th���䣬��ֵΪ�ֵ������ֵ֮��ļ�Ȩ��
                    input(h_index, w_index, 1) = (5/9)*ch_high_value_r / 255 + (4/9)*ch_low_value_r / 255;
                    input(h_index, w_index, 2) = (5/9)*ch_high_value_g / 255 + (4/9)*ch_low_value_g / 255;
                    input(h_index, w_index, 3) = (5/9)*ch_high_value_b / 255 + (4/9)*ch_low_value_b / 255;
                    
                elseif (bdr_pix_val_min + 5*bdr_interval <= patch_gray(h_index, w_index)) && ...
                        (patch_gray(h_index, w_index) < bdr_pix_val_min + 6*bdr_interval)
                    % �߽�6th���䣬��ֵΪ�ֵ������ֵ֮��ļ�Ȩ��
                    input(h_index, w_index, 1) = (6/9)*ch_high_value_r / 255 + (3/9)*ch_low_value_r / 255;
                    input(h_index, w_index, 2) = (6/9)*ch_high_value_g / 255 + (3/9)*ch_low_value_g / 255;
                    input(h_index, w_index, 3) = (6/9)*ch_high_value_b / 255 + (3/9)*ch_low_value_b / 255;
                    
                elseif (bdr_pix_val_min + 6*bdr_interval <= patch_gray(h_index, w_index)) && ...
                        (patch_gray(h_index, w_index) < bdr_pix_val_min + 7*bdr_interval)
                    % �߽�7th���䣬��ֵΪ�ֵ������ֵ֮��ļ�Ȩ��
                    input(h_index, w_index, 1) = (7/9)*ch_high_value_r / 255 + (2/9)*ch_low_value_r / 255;
                    input(h_index, w_index, 2) = (7/9)*ch_high_value_g / 255 + (2/9)*ch_low_value_g / 255;
                    input(h_index, w_index, 3) = (7/9)*ch_high_value_b / 255 + (2/9)*ch_low_value_b / 255;
                    
                elseif (bdr_pix_val_min + 7*bdr_interval <= patch_gray(h_index, w_index)) && ...
                        (patch_gray(h_index, w_index) <= bdr_pix_val_min + 8*bdr_interval)
                    % �߽�8th���䣬��ֵΪ�ֵ������ֵ֮��ļ�Ȩ��
                    input(h_index, w_index, 1) = (8/9)*ch_high_value_r / 255 + (1/9)*ch_low_value_r / 255;
                    input(h_index, w_index, 2) = (8/9)*ch_high_value_g / 255 + (1/9)*ch_low_value_g / 255;
                    input(h_index, w_index, 3) = (8/9)*ch_high_value_b / 255 + (1/9)*ch_low_value_b / 255;
                    
                end       
            end
        end
    end
end
end