function [ result ] = svmpredict( models,test )

numClasses=10;
result = zeros(size(test,1),1);
for j=1:size(test,1)
    for k=1:numClasses
        if(svmclassify(models(k),test(j,:))) 
            result(j) = k-1;
        end
    end
    %real lable in num recognization are 0 to 9, so minus 1.
end

end

