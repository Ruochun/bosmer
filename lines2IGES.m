function lines2IGES(linesData, OutName, BaseDir)

%     All the spacing is done manually so if any of the G section words
%     are changed the spacing will need to be fixed.

G = {}; D = {}; P = {};
nam = 'LinIGES_Output';
       
%**************************information*********************************
SLine = 'Matlab Nurbs converted -> IGES file.    				 ';
%GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
G{1,1} = ',';      % Parameter Deliminator Character
G{2,1} = ';';      % Record Delimiter Character
G{3,1} = nam;      % Product ID from Sender
G{4,1} = OutName;   % File Name
G{5,1} = 'MatLin'; % System ID
G{6,1} = 'Matlab -> IGES v3.0 Aug 2006';            % Pre-processor Version
G{7,1} = 16;       % Number of Bits for Integers
G{8,1} = 6;        % Single Precision Magnitude
G{9,1} = 15;       % Single Precision Significance
G{10,1}= 13;       % Double Precision Magnitude
G{11,1}= 15;       % Double Precision Significance
G{12,1}= nam;      % Product ID for Receiver
G{13,1}= 1.00000;  % Model Space Scale
G{14,1}= 3;        % Unit Flag (1 = inches, 3 = look to index 15 name)
G{15,1}= 'MM';     % Units  (Inches = "INCH")
G{16,1}= 8;        % Maximum Number of Line Weights
G{17,1}= 0.0160000; % Size of Maximum Line Width
G{18,1}= 'Today';   % Date and Time Stamp  ** fix me **
G{19,1}= 0.000000001; % Minimum User-intended Resolution
G{20,1}= 300.000;   % Approximate Maximum Coordinate
G{21,1}= 'Ruochun    Zhang: ruochunz@gmail.com';     % Name of Author
G{22,1}= 'UWMadison ME Dept. ';     % Author's Organization
G{23,1}= 11; % - USPRO/IPO-100 (IGES 5.2) [USPRO93]';  % IGES Version Number  ** prob not right **
G{24,1}= 3; % - ANSI';            % Drafting Standard Code
G{25,1}= 'Today of course'; % Model Creation/Change Date


%% pre-processing of the input
numLoops = size(linesData,1);
iterSeg = 0;
for i=1:numLoops
    loopI = linesData{i};
    numSegs = floor(size(loopI,2)/2);
    for j=1:numSegs
        
        Z = 0.0;
        X1 = loopI(2*j-1); Y1 = loopI(2*j);
        if j<numSegs
            X2 = loopI(2*j+1); Y2 = loopI(2*j+2);
        else
            X2 = loopI(1); Y2 = loopI(2);
        end
%DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDd

        iterSeg = iterSeg + 1;     
        numLine = 1;

        D{1,iterSeg} = 110;       % Entity Type.
        D{2,iterSeg} = iterSeg; % Data Start line
        D{3,iterSeg} = 0;         % Structure
        D{4,iterSeg} = 1;         % Line Font Pattern (1= Solid)
        D{5,iterSeg} = 0;         % Level
        D{6,iterSeg} = 0;         % View
        D{7,iterSeg} = 0;         % Transformation Matrix
        D{8,iterSeg} = 0;         % Label Display
        D{9,iterSeg} = 0;         % Blank Status (0 = Visible)
        D{10,iterSeg}= 0;         % Subord. Entity Switch (0 = Independant)
        D{11,iterSeg}= 0;         % Entity Use Flag (0 = Geometry)
        D{12,iterSeg}= 0;         % Hierarchy ( 1 = Global defer)
        D{13,iterSeg}= 2;         % Line Weight Number
        D{14,iterSeg}= numLine;   % How many P lines this object has
        D{15,iterSeg}= 0;         % Form Number (9 = General Quadratic Surface), 0 = none of above (1-9) options

%PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
        P{1,iterSeg} = 110;                   % Entity Type. 
        P{2,iterSeg} = X1;
        P{3,iterSeg} = Y1;
        P{4,iterSeg} = Z;
        P{5,iterSeg} = X2;
        P{6,iterSeg} = Y2;
        P{7,iterSeg} = Z;

    end
end

%************************Data export to file*******************************
% All lines must have exactly 80 characters across
filnam = [BaseDir OutName];
fid=fopen(filnam, 'w');

%------S Line information (The initial line to get things started)---------
% fprintf(fid,
% '*********|*********|*********|*********|*********|*********|*********|*********|\n');  80 Characters for measurment
Sln =1;
fprintf(fid, '%s                S%07.f\n', SLine, Sln);

%---------------G Line information  (Header infomation)--------------------
Gln=1;
for i = 1:25  % Number of characters after each one
    tmp = str2mat(G{i,1});
    len=size(tmp);
    G{i,2} = len(1,2);
    G{i,3} = ischar(G{i,1});  % 1 = yes, 0 = no
end 
% case1
% fprintf(fid, '%1.0fH%s,', G{1,2}, G{1,1});              % 1            
% fprintf(fid, '%1.0fH%s,', G{2,2}, G{2,1});   
% fprintf(fid, '%2.0fH%s,', G{3,2}, G{3,1});  
% fprintf(fid, '%2.0fH%s,                               G%07.f\n', G{4,2}, G{4,1}, Gln);  Gln=Gln+1;
% case 2
% the following code replaces the original code from line 152-155
tstr = sprintf('%1.0fH%s,', G{1,2}, G{1,1}); 
tstr = [tstr,sprintf('%1.0fH%s,', G{2,2}, G{2,1})]; 
tstr = [tstr,sprintf('%2.0fH%s,', G{3,2}, G{3,1})]; 
tstr = [tstr,sprintf('%2.0fH%s,', G{4,2}, G{4,1})]; 
fprintf(fid, '%s',tstr); 
len_tstr = length(tstr); 
for kk = 1:(72-len_tstr) 
fprintf(fid, ' '); 
end 
fprintf(fid, 'G%07.f\n', Gln); Gln=Gln+1;
%%%%%%%%%%%%%%%%%%%%%%
fprintf(fid, '%1.0fH%s,', G{5,2}, G{5,1});              % 5
fprintf(fid, '%2.0fH%s,', G{6,2}, G{6,1});
fprintf(fid, '%2.0f,',    G{7,1});
fprintf(fid, '%2.0f,',    G{8,1}); 
fprintf(fid, '%2.0f,',    G{9,1});
fprintf(fid, '%2.0f,',    G{10,1});                    % 10
fprintf(fid, '%2.0f,                G%07.f\n', G{11,1}, Gln);  Gln=Gln+1;
fprintf(fid, '%2.0fH%s,', G{12,2}, G{12,1});
fprintf(fid, '%5.2f,',    G{13,1});   
fprintf(fid, '%2.0f,',    G{14,1});     
fprintf(fid, '%2.0fH%s,', G{15,2}, G{15,1});            % 15
fprintf(fid, '%2.0f,',    G{16,1});      
fprintf(fid, '%7.4f,',    G{17,1});      
fprintf(fid, '%2.0fH%s,', G{18,2}, G{18,1});
fprintf(fid, '%5.5f,           G%07.f\n', G{19,1},  Gln); Gln=Gln+1;
fprintf(fid, '%2.0f,',    G{20,1});                     % 20
fprintf(fid, '%2.0fH%s,                            G%07.f\n', G{21,2}, G{21,1}, Gln); Gln=Gln+1;
fprintf(fid, '%2.0fH%s,                                                 G%07.f\n', G{22,2}, G{22,1}, Gln); Gln=Gln+1;
fprintf(fid, '%2.0f,',    G{23,1});
fprintf(fid, '%2.0f,',    G{24,1});
fprintf(fid, '%2.0fH%s;                                               G%07.f\n', G{25,2}, G{25,1}, Gln);

%------------------D Line information (Data information)-------------------

for Dln=1:iterSeg
% Doing this with only one entity... 
    fprintf(fid, '%8.0f%8.0f%8.0f%8.0f%8.0f%8.0f%8.0f               1D%07.f\n',...
             D{1,Dln}, D{2,Dln}, D{3,Dln}, D{4,Dln}, D{5,Dln}, D{6,Dln}, D{7,Dln}, 2*Dln-1);
    fprintf(fid, '%8.0f%8.0f%8.0f%8.0f%8.0f                                D%07.f\n',...
             D{1,Dln},D{12,Dln},D{13,Dln},D{14,Dln},D{15,Dln}, 2*Dln);
end
%-----------------P Line information  (All the data)-----------------------
for Pln=1:iterSeg
    fprintf(fid, '%3.0f,%09.5f,%09.5f,%09.5f,%09.5f,%09.5f,%09.5f; %07.fP%07.f\n',...
            P{1,Pln},   P{2,Pln},  P{3,Pln},  P{4,Pln},  P{5,Pln}, P{6,Pln},   P{7,Pln}, 2*Pln-1,Pln); 
end

%-----------------T Line information  (Termination)------------------------
fprintf(fid, 'S%07.fG%07.fD%07.fP%07.f                                        T0000001',...
             Sln, Gln, 2*Dln, Pln);
fclose(fid);

'Finished Export to IGES'
end