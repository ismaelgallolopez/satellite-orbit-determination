function load_directory_data(dirname, suffix)
    if nargin < 2
        suffix = '';
    end
    files = dir(fullfile(dirname, '*.txt'));
    for i = 1:length(files)
        fname = files(i).name;
        [~, fileName, ~] = fileparts(fname);
        varName = [fileName, suffix];
        data = load(fullfile(dirname, fname));
        assignin('caller', varName, data);
    end
end
