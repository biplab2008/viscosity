%% copy files randomly
copy_files = false;
path = 'D:\All_files\pys\AI_algos\Mikes_Work\viscosity-video-classification\code_digdiscovery\annotate\';
file_path = 'D:\All_files\pys\AI_algos\Mikes_Work\viscosity-video-classification\code_digdiscovery\new_honey_164\';
cropped_path = [path, 'cropped\'];

tot_frames = 1;
folders = dir(file_path);
if copy_files
    for i = 1: length(folders)

        if ~strcmp(folders(i).name,'.') && ~strcmp(folders(i).name,'..')
            frame_list = randperm(30,tot_frames);
            for j = 1 : tot_frames
                frames = ['frame_',num2str(frame_list(j)),'.jpg'];
                copyfile([file_path,folders(i).name,'\',frames],[path,folders(i).name,'_',frames]);
            end
        end

    end
end


%% check a single image and crop the rest using the smale BBox
files = dir([path,'*.jpg']);
% figure;
% file = '2202_frame_6.jpg';
% img = imread([path,file]);subplot(1,2,1);imshow(img);
% h = drawrectangle();
% img_crop = imcrop(img,h.Position);
% subplot(1,2,2);imshow(img_crop);


for i = 1: length(files)

    if ~strcmp(files(i).name,'.') && ~strcmp(files(i).name,'..')
        
        img = imread([path,files(i).name]);
        img_crop = imcrop(img,h.Position);
        imwrite(img_crop,[cropped_path,files(i).name])

    end
end

