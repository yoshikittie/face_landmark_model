name_list = dir('HELEN/*.mat');
% name
% date
% bytes
% isdir

l = length(name_list);
txt_out = fopen('HELEN.txt', 'a');
for i = 1 : l
    file_name{i} = name_list(i).name;
    path = strcat('HELEN/', file_name{i});
    % disp(path);
    mat = load(path);
    landmarks = mat.pt2d;
    fprintf(txt_out, '%s\n', path);
    % disp(size(landmarks));
    for j = 1 : 68
        fprintf(txt_out, '%d %d\n', round(landmarks(1, j)), round(landmarks(2, j)));
    end
end
fclose(txt_out);