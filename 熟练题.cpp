void traversal(TreeNode *root, vector<int> &res){
    if(root==NULL) return;
    traversal(root->left);
    res.push_back(root->val);
    traversal(root->right);
}
bool isValidBST(TreeNode *root){
    vector<int> vec;
    traversal(root, vec);
    for(int i=1;i<vec.size();i++){
        if(vec[i]<=vec[i-1])
            return false;
    }
    return true;
}


bool isValidBST(TreeNode *root){
    if(root== NULL) return true;
    vector<int> res;
    stack<TreeNode *> s;
    TreeNode *cur = root;
    while(!s.empty() || cur != NULL){
        if(cur != NULL){
            s.push(cur);
            cur=cur->left;
        }else{
            cur = s.top();
            s.pop();

            res.push(cur->val);
            cur=cur->right;
        }
    }
    for(Int i=0;i<res.size()-1;i++){
        if(res[i] > res[i+1])
            return false;
    }
    return true;
    
}

int getmin(TreeNode *root){
    stack<TreeNode *> s;
    if(root==NULL) return -1;
    TreeNode *cur=root;
    TreeNode *pre=NULL;
    int res=INT_MAX;
    while(!s.empty() || cur != NULL){
        if(cur != NULL){
            s.push(cur);
            cur=cur->left;
        }else{
            cur=s.top();
            s.pop();
            if(pre !=NULL){
                res=min(res, cur->val - pre->val);
            }
            pre=cur;
            cur=cur->right;
        }
    }
    return res;
}

vector<int> findMode(TreeNode *root){
    vector<int> res;
    stack<TreeNode *> s;
    TreeNode *cur = root;
    TreeNode *pre = NULL;
    int count;
    int maxCount = 0;
    while(!s.empty() || cur != NULL){
        if(cur != NULL){
            s.push(cur);
            cur = cur->left;    //左
        }else{
            cur = s.top();      //中
            s.pop();
            if(pre == NULL) 
                count=1;
            else if(pre->val == cur->val){
                count++;
            }else{
                count=1;
            }
            if(count == maxCount){
                res.push_back(cur->val);
            }
            if(count > maxCount){
                maxCount = count;
                res.clear();
                res.push_back(cur->val);
            }
            pre=cur;
            cur=cur->right;         //右
        }
    }
    return res;
}


TreeNode*  lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q){
    if(root == NULL || root == p || root == q) return root;
    TreeNode *left = lowestCommonAncestor(root->left, p, q);
    TreeNode *right = lowestCommonAncestor(root->right, p, q);
    if(left != NULL && right != NULL) return root;
    
    if(left == NULL && right != NULL) return right;
    else if(left != NULL && right == NULL) return left;
    else return NULL;
}

h(0)=1
h(1)=1
h(2)=h(0)h(1)+h(1)h(0)=2
h(3)=h(0)h(2)+h(1)h(1)+h(2)h(0)=5
h(n)=h(0)h(n-1)+h(1)h(n-2)+...+h(n-2)h(1)+h(n-1)h(0)

int numTree(int n){
    vector<int> dp(n+1);
    dp[0]=1, dp[1]=1;
    for(int i = 2;i <= n;i++){
        int sum = 0;
        for(j = 0; j <= i-1; j++){
            sum += dp[j]*dp[i-j-1];
        }
        dp[i]=sum;
    }
    return dp[n];
}

int path(int m, int n){
    vector<vector<int>> dp(m, vector<int>(n,0);
    for(int i=0;i<m && obj[i][0]==0;i++) dp[i][0]=1;
    for(int j=0;j<n && obj[0][j]==0;j++) dp[0][j]=1;

    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if(obj[i][j]==1)
                continue;
            dp[i][j]=dp[i-1][j]+dp[i]
        }
    }
    dp[m-1][n-1];
}

vector<int> dp(n+1);
dp[2]=1;
for(int i=3;i<=n;i++){
    for(int j=1;j<i;j++){
        dp[i]=max(i)
    }
    
}



dp[0]=1
dp[1]=1
dp[2]=dp[0]dp[1]+dp[1]dp[0]=2
dp[3]=dp[0]dp[2]+dp[1]dp[1]+dp[2]dp[0]=2+1+2=5

dp[0]=1
dp[1]=1

int path(int n){
    if(n==0 || n==1) return n;
    vector<int> dp(n+1);
    for(int i=2;i<=n;i++){
        for(int j=0;j<=i-1;j++){
            dp[i]  += dp[j]dp[i-j-1];
        }
    }
    return dp[n];
}


int m=s1.size(),n=s2.size();
if(m+n != s3.size()) return false;

vector<vector<int>> dp(m+1, vector<int>(n+1, false));
dp[0][0]=true;
for(int i=1;i<=m;i++){
    for(int j=1;j<=n;j++){
        if(i-1 >= 0)
            dp[i][j]=dp[i][j] || (dp[i-1][j] && s1[i-1]==s3[i+j-1])
        if(j-1 >= 0)
            dp[i][j]=dp[i][j] || (dp[i][j-1] && s2[j-1]==s3[i+j-1]);
    
    }
}
0-1
sum/2
nums,

dp[j]=max(dp[j], dp[j-weight[i]]+value[i]);

bool canpartition(vector<int> &nums){
    int sum=0;
    for(int i=0;i<nums.size();i++)
        sum+=nums[i];
    if(sum%2!=0) return false;
    int bag=sum/2;
    vector<int> dp(10001, 0);

    for(int i=0;i<nums.size();i++){
        for(int j=bag;j>=nums[i];j--){
            dp[j]=max(dp[j], dp[j-nums[i]]+nums[i]);
        }
    }
    if(dp[bag]==bag) return true;
    return false; 
}

 

int lastStoneWeightII(vector<int>& stones) {
    int sum=0;
    for(auto a:stones) sum += a;
    int target=sum/2;

    vector<int> dp(15001,0);
    for(int i=0;i<stores.size();i++){  //物品
        for(int j=target;j>=stores[i];j--){  //背包
            dp[j]=max(dp[j], dp[j-stores[i]]+stores[i]);
        }
    }
    return sum-dp[target]-dp[target];
}

left + right = sum
left -( sum - left) = target
left = (target + sum) / 

vector<vector<int>> res;
vector<int> path;
void traversal(vector<int> candidates, int &startidx, int &target, int &sum){
    if(sum == target){
        res.push(path);
        return;
    }
    if(sum > target)
        return;

    for(int i=startidx;i<candidates.size();i++){
        sum+=candidates[i];
        path.push_back(candidates[i]);
        traversal(candidates, i, target, sum);
        path.pop_back();
        sum-=candidates[i];
    }
}

vector<vector<int>> combinationSum(vector<int> candidates, int target){
    res.clear();
    path.clear();
    traversal(candidates, 0, target, 0);
}

vector<vector<int>> res;
vector<int> path;
void traversal(int &n, int &k, int startidx){
    if(path.size() == k){

    }
}
vector<vector<int>> zuheOne(int n, int k){
    res.clear();
    path.clear();
    traversal(n, 0, k)
}


组合1：
1 2 3 4

1    2  3   4
234  34  4
12 13 14  23 24  34  
vector<vector<int>> res;
vector<int> path;
void traversal(int &n, int &k, int startidx){
    if(path.size() == k){
        res.push_back(path);
        return; 
    }
    for(int i=startidx;i<=n;i++){
        path.push_back(i);
        traversal(n,k,i+1);
        path.pop_back();
    }
}
vector<vector<int>> allCan(int n, int k){
    res.clear();
    path.clear();
    traversal(n, k, 1);
    return res;
}

组合3
1 2 3 4 5 6 7 8 9
23456789

vector<vector<int>> res;
vector<int> path;
void traversal(int &n, int &k, int startidx, int sum){
    

    if(path.size() == k){
        if(sum == n)
            res.push_back(path);
        return;
    }

    for(int i=startidx;i<=9;i++){
        path.push_back(i);
        sum += i;
        traversal(n, k, i+1, sum);
        sum -= i;
        path.pop_back();
    }
}
vector<vector<int>> allCan(int n, int k){
    res.clear();
    path.clear();
    traversal(n, k, 1, 0);
}


电话号码字母组合
vector<string> res;
string s;
string letter[10] = {"", "","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz",};
    
void traversal(const string &dig, int &idx){
    if(idx==dig.size()){
        res.push_back(s);
        return;
    }
    int digit = dig[idx] - '0';
    string let = letter[digit];
    for(int i=0;i<let.size();i++){
        s.push_back(let[i]);
        traversal(dig, i+1);
        s.pop_back();
    }
}
vector<string> allzuhe(string s){
    if(s.size() ==0) return res;
    traversal(s, 0);
    return res;
}



组合总和1
vector<vector<int>> res;
vector<int> path;
void traversal(vector<int> &candidate, int &target, int &idx, int &sum){
    if(sum > target)    return;
    if(sum == target){
        res.push_back(path);
        return;
    }

    for(int i=idx, i<candidate.size();i++){
        path.push_back(candidate[i]);
        sum += candidate[i];
        traversal(candidate, target, i+1, sum);
        path.pop_back();
        sum -= candidate[i];
    }
}
vector<vector<int>> zuhe(vector<int> candidate, int target){
    traversal(candidate, target, 0, 0);
    return res;
}

组合总和2
vector<vector<int>> res;
vector<int> path;
void traversal(vector<int> &candidates, int target, int startidx, int sum){
    if(sum > target) return;
    if(sum == target){
        res.push_back(path);
        return;
    }
    for(int i=startidx;i<candidates.size();i++){
        sum += candidates[i];
        path.push_back(candidates[i]);
        traversal(candidates,target,i,sum);
        sum -= candidates[i];
        path.pop_back();

    }

}
vector<vector<int>> allPath(vector<int> candidates, int target){
    void traversal(candidates, target, 0, 0);
}

组合总和2
vector<vector<int>> res;
vector<int> path;
void traversal(vector<int> &candidates, int target, int startidx, int sum){
    if(sum > target) return;
    if(sum == target){
        res.push_back(path);
        return;
    }
    //同一数层上不能重复
    for(int i=startidx;i<candidates.size() && sum + candidates[i]<=target;i++){
        if(i>= && candidates[i]==candidates[i-1]&&used[i-1]==false)
            continue;
        sum += candidates[i];
        path.push_back(candidates[i]);
        used[i]=true;
        traversal(candidates,target,i,sum);
        used[i]=false;
        sum -= candidates[i];
        path.pop_back();

    }

}
vector<vector<int>> allPath(vector<int> candidates, int target){
    void traversal(candidates, target, 0, 0);
}

回文串
void isPalindrome(string s, int start,int end){
    for(int i=start;j=end;i<j;i++,j--){
        if(s[i] != s[j])    return false;
        else return true;
    }
}
vector<vector<string>> res;
vector<string> path;
void traversal(string s, int startidx){
    if(startidx>=s.size())
    {
        res.push_back(path);
        return;
    }
    for(int i=startidx;i<s.size();i++){
        if(isPalindrome(s,startidx,i)){
            string str = s.substr(startidx,i-startidx+1);
            path.push_back(str);
        }else
            continue;
        traversal(s,i+1);
        path.pop_back();
    }
}
vector<vector<string>> all(string s){
    traversal(s, 0);
}


组合，切割，全排列，子集

子集
vector<vector<int> res;
vector<int> path;
void traversal(vector<int>& nums,int startidx){
    res.push_back(path);
    if(startidx>=nums.size()){
        return;
    }
    for(int i=startidx;i<nums.size();i++){
        path.push_back(nums[i]);
        traversal(nums,i+1);
        path.pop_back();
    }
}
vector<vector<int> subsets(vector<int>& nums){
    traversal(nums,startidx)
    return res;
} 

子集去重
vector<vector<int> res;
vector<int> path;
void traversal(vector<int>& nums,int startidx,vector<bool>& used){
    res.push_back(path);
    if(startidx>=nums.size()){
        return;
    }
    for(int i=startidx;i<nums.size();i++){
        if(nums[i] == nums[i-1] && used[i]==false) //同一层
            continue
        used[i]=true;
        path.push_back(nums[i]);
        traversal(nums,i+1);
        path.pop_back();
        used[i]=false;
    }
}
vector<vector<int> subsets(vector<int>& nums){
    vector<bool> used(nums.size(), false);
    sort(nums.begin(), nums.end());
    traversal(nums,startidx, used)
    return res;
} 

递增子序列
vector<vector<int>> res;
vector<int> path;
1.不能排序，同一层去重
2.每个path最少2个值
3.递增
void traversal(vector<int>& nums,int startidx){
    if(path.size() >=2){
        res.push_back(path);
        return;
    }
    unordered_set<int> uset;
    for(int i=startidx;i<nums.size();i++){
        if((!path.empty() && path.back()>nums[i]) || uset.find(nums[i])!=uset.end())
            continue;
        uset.insert(nums[i]);
        path.push_back(nums[i]);
        traversal(nums,i+1);
        path.pop_back();

    }

    

}
vector<vector<int> subxulies(vector<int>& nums){

    traversal(nums, startidx);
    return res;
}

全排列  无重复元素
1.{1,2} {2,1}是两个集合
2.设定used确定那个元素使用过了那个么有使用过

vector<vector<int>> res;
vector<int> path;
void traversal(vector<int>& nums, vector<bool> &used){
    if(path.size() == nums.size()){
        res.push_back(path);
        return;
    }

    for(int i=0;i<nums.size();i++){
        if(used[i]==true)
            continue;
        path.push_back(nums[i]);
        used[i]=true;
        traversal(nums, used);
        path.pop_back();
        used[i]=false;
    }
}
vector<vector<int> quanpailie(vector<int>& nums){
    vector<bool> used(nums.size(), false);
    traversal(nums,used);
    return res;
}

全排列2  有重复元素
vector<vector<int>> res;
vector<int> path;
void traversal(vector<int>& nums, vector<bool> &used){
    if(nums.size() == path.size()){
        res.push_back(path);
        return;
    }

    for(int i=0;i<nums.size();i++){
        if(i>0 && nums[i]==nums[i-1] && used[i-1]==false)
            continue;
        if(used[i]==true)
            continue;
        path.push_back(nums[i]);
        used[i]=true;
        traversal(nums, used);
        path.pop_back();
        used[i]=false;
    }
}

vector<vector<int> quanpailie2(vector<int>& nums){
    sort(nums.begin(), nums.end());
    vector<bool> used(nums.size(), false);
    traversal(nums,used);
    return res;
}

N皇后
vector<vector<string>> res;
vector<string> path;
bool isValid(int row, int col,vector<string> &chess, int n){
    for(int i=0;i<row;i++){
        if(chess[i][col]=='Q')
            return false;
    }
    for(int i=row-1;j=col-1;i>=0&&j>=0;i--,j--){
        if(chess[i][j]=='Q')
            return false;
    }
    for(int i=row+1;j=col+1;i<n&&j<n;i++,j++){
        if(chess[i][j]=='Q')
            return false;
    }
    return true;
}
void traversal(int n, int row){
    if(row=n){
        res.push_back(path);
        return;
    }
    for(int col=0;col<n;col++){
        if(isValid(row,col,path,n)){
            path[row][col]='Q';
            traversal(n,row+1);
            path[row][col]='.'
        }
    }
}
vector<vector<int> QQueen(int n){
    traversal(n,1);
    return res;
}

_______________________________________
组合总和1
vector<vector<int>> res;
vector<int> path;
void traversal(int k, int start, int target, int sum){
    if(sum > target) return;
    if(k == path.size()){
        if(sum == target)
            res.push_back(path);
        return;
    }
    for(int i=start;i<=9;i++){
        path.push_back(i);
        sum += i;
        traversal(k,i+1,target,sum);
        sum -= i;
        path.pop_back();
    }
}
vector<vector<int>> zuhe1(int k, int n){
    traversal(k,1,n,0);
    return res;
}

组合总和2:集合元素会有重复
vector<vector<int>> res;
vector<int> path;
void traversal(vector<int> &candidate int target, int start, int sum){
    if(sum > target) return;
    if(sum == target)
        res.push_back(path);
        return;
    }
    for(int i=start;i<candidate.size();i++){
        path.push_back(candidate[i]);
        sum += candidate[i];
        traversal(k,i,target,sum);
        sum -= candidate[i];
        path.pop_back();
    }
}
vector<vector<int>> zuhe2(vector<int> &candidate, int target){
    traversal(candidate, target, 0, 0);
    return res;
}


组合总和3:集合元素会有重复，但要求解集不能包含重复的组合。
vector<vector<int>> res;
vector<int> path;
void traversal(vector<int> &candidate, int target, int start, vector<int> &used, int sum){
    if(sum > target) return;
    if(sum == target)
        res.push_back(path);
        return;
    }
    for(int i=start;i<candidate.size() && sum+candidate[i]<=target;i++){
        if(candidate[i]==candidate[i-1] && used[i-1]=false) 
            continue;
        used[i]=true;
        path.push_back(candidate[i]);
        sum += candidate[i];
        traversal(candidate,target,i+1,used,sum);
        sum -= candidate[i];
        path.pop_back();
        used[i]=false;
    }
}
vector<vector<int>> zuhe2(vector<int> &candidate, int target){
    sort(candidate.begin(), candidate.end());
    vector<bool> used(candidate.size(), false);
    traversal(candidate, target, 0, used, 0);
    return res;
}
子集1
vector<vector<int>> res;
vector<int> path;
void traversal(vector<int> &candidate, int start){
    res.push_back(path);
    if(start >= candidate.size())
        return;

    for(int i=start;i<candidate.size();i++){
        path.push_back(candidate[i]);
        traversal(candidate,i+1);
        path.pop_back();
    }
}
vector<vector<int>> ziji1(vector<int> &candidate){
    traversal(candidate, 0);
    return res;
}

子集2:可能包含重复元素,解集不能包含重复的子集。
vector<vector<int>> res;
vector<int> path;
void traversal(vector<int> &candidate, int start,vector<bool> &used){
    res.push_back(path);
    if(start >= candidate.size())
        return;
    }

    for(int i=start;i<candidate.size();i++){
        if(i > 0 && candidate[i]==candidate[i-1] && used[i-1]=false) 
            continue;
        used[i]=true;
        path.push_back(candidate[i]);
        traversal(candidate,i+1,used);
        path.pop_back();
        used[i]=false;
    }
}
vector<vector<int>> jieji2(vector<int> &candidate){
    sort(candidate.begin(), candidate.end());
    vector<bool> used(candidate.size(), false);
    traversal(candidate, 0, used);
    return res;
}

递增子序列 [4767]:所有该数组的递增子序列
vector<vector<int>> res;
vector<int> path;
void traversal(vector<int> &candidate, int start){
    res.push_back(path);
    if(start >= candidate.size())
        return;
    }
    unordered_set<int> uset;
    for(int i=start;i<candidate.size();i++){
        
        if((!path.empty() && candidate[i] < path.back()) || uset.find(candidate[i])!=uset.end())
            continue;
        uset.insert(candidate[i]);
        path.push_back(candidate[i]);
        traversal(candidate,i+1,used);
        path.pop_back();

    }
}
vector<vector<int>> jieji2(vector<int> &candidate){
    traversal(candidate, 0);
    return res;
}

排列1
vector<vector<int>> res;
vector<int> path;
void traversal(vector<int> &candidate, vector<bool> used){
    if(path.size() == candidate.size()){
        res.push_back(path);
        return;
    }

    for(int i=0;i<candidate.size();i++){
        
        if(used[i]==true)
            continue;
        used[i]=true;
        path.push_back(candidate[i]);
        traversal(candidate,used);
        path.pop_back();
        used[i]=false;
    }
}
vector<vector<int>> quanpailie(vector<int> &candidate){
    vector<bool> used(candidate.size(), false);
    traversal(candidate, used);
    return res;
}

排列2:可包含重复数字的序列，要返回所有不重复的全排列。
vector<vector<int>> res;
vector<int> path;
void traversal(vector<int> &candidate, vector<bool> used){
    if(path.size() == candidate.size()){
        res.push_back(path);
        return;
    }

    for(int i=0;i<candidate.size();i++){
        if(i>0 && candidate[i]==candidate[i-1] && used[i-1]==false)
            continue
       
        used[i]=true;
        path.push_back(candidate[i]);
        traversal(candidate,used);
        path.pop_back();
        used[i]=false;
    }
}
vector<vector<int>> quanpailie(vector<int> &candidate){
    sort(candidate.begin(), candidate.end());
    vector<bool> used(candidate.size(), false);
    traversal(candidate, used);
    return res;
}

动态规划
基础问题——
int fib(int n){
    if(n<=1) return n;
    vector<int> dp(n+1);
    dp[0]=0;
    dp[1]=1;
    for(int i=2;i<=n;i++){
        dp[i]=dp[i-1]+dp[i-2];
    }
    return dp[n];
}

int fib(int n){
    if(n<=1) return n;
    int dp[2];
    dp[0]=0;
    dp[1]=1;
    for(int i=2;i<=n;i++){
        int sum=dp[i-1]+dp[i-2];
        dp[0]=dp[1];
        dp[1]=sum;
    }
    return dp[1];
}
你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
int palouti(int n){
    if(n<=1) return n;
    vector<int> dp(n+1);
    dp[1]=1;
    dp[2]=2;
    for(int i=3;i<=n;i++){
        dp[i]=dp[i-1]+dp[i-2];
    }
    return dp[n];
}
int palouti(int n){
    if(n<=1) return n;
    int dp[3];
    dp[1]=1;
    dp[2]=2;
    for(int i=3;i<=n;i++){
        int sum=dp[i-1]+dp[i-2];
        dp[1]=dp[2];
        dp[2]=sum;
    }
    return dp[2];
}
花费爬楼梯
int costPalouti(vector<int> &cost){
    vector<int> dp(cost.size());
    dp[0]=cost[0];
    dp[1]=cost[1];
    for(int i=2;i<cost.size();i++){
        dp[i]=min(dp[i-1], dp[i-2])+cost[i];
    }
    return min(dp[cost.size()-1], dp[cost.size()-2]);
}
机器人
int path(int m, int n){
    vector<vector<int>> dp(m, vector<int>(n, 0));
    for(int i=0;i<m;i++)
        dp[i][0]=1;
    for(int j=0;j<n;j++)
        dp[0][j]=1;
    for(int i=1;i<m;i++){
        for(int j=1;j<n;j++){
            dp[i][j]=dp[i-1][j]+dp[i][j-1];
        }
    }
    return dp[m-1][n-1];
}

int path2(int[][] obstacleGrid){
    int m=obstacleGrid.size(), n=obstacleGrid[0].size();
    vector<vector<int>> dp(m, vector<int>(n, 0));
    for(int i=0;i<m && obstacleGrid[i][0]!=1;i++)
        dp[i][0]=1;
    for(int j=0;j<n && obstacleGrid[0][j]!=1;j++)
        dp[0][j]=1;
    for(int i=1;i<m;i++){
        for(int j=1;j<n;j++){
            if(obstacleGrid[i][j] == 1)
                continue;
            dp[i][j]=dp[i-1][j]+dp[i][j-1];
        }
    }
    return dp[m-1][n-1];
}
整数拆分
int chaifen(int n){
    vector<int> dp(n+1, 0);
    dp[1]=1;
    dp[2]=1;
    for(int i=3;i<=n;i++){
        for(int j=1;j<i-1;j++){
            dp[i]=max(j*(i-j), j*dp[i-j])
        }
    }
    return dp[n];
}

dp[2]=dp[0]dp[1]+dp[1]dp[0]
dp[3]=dp[0]dp[2]+dp[1]dp[1]+dp[2]dp[0];
dp[4]=dp[0]dp[3]+dp[1]dp[2]+dp[2]dp[1]+dp[3]dp[0]
int erchaBst(int n){
    if(n<=2) return n;
    vector<int> dp(n+1, 0);
    dp[0]=1;
    dp[1]=1;
    dp[2]=2;
    for(int i=3;i<=n;i++){
        for(int j=0;j<i;j++){
            dp[i] += dp[j]*dp[i-j-1];
        }
    }
}

物品
背包
价值
二维数组
dp[i][j]=max(dp[i-1][j], dp[i-1][j-weight[i]]+value[i])
dp[i][0]=0;
for(int j=weight[0];j<=bagweight;j++)
    dp[0][j]=value[0];

dp[0][j]=value[0]

物品0	1	15
物品1	3	20
物品2	4	30
      0  1  2  3  4
物品0 0  15 15 15 15    
物品1 0  15 15 20 35 
物品2 0  15 15 20 35  

分割等和子集
vector<int> fenge(vector<int> &nums){
    vector<int> dp(10001, 0);
    int sum;
    for(int i=0;i<nums.size();i++)
        sum += nums[i];
    if(sum % 2 != 0) return false;
    int bag = sum / 2;

    
    for(int i=0;i<nums.size();i++){ //物品
        for(int j=bag;j>=nums[i];j--)
            dp[j]=max(dp[j], dp[j-nums[i]]+nums[i]);
        
    }
    if(dp[bag]==bag)
        return true;
    return false;
}

int lastStone(vector<int>  &stones){
    vector<int> dp(15001, 0);
    int sum = 0;
    for(int i=0;i<stones.size();i++)
        sum += stones[i];
    int target=sum/2;
    for(int i=0;i<stones.size();i++){
        for(int j=target;j>=stones[i];j--){
            dp[j]=max(dp[j],dp[j-stones[i]]+stones[i]);
        }
    }
    return sum-dp[target]-dp[target];
}

left-right=target
left-(sum-left)=target
2left-sum=target
left=(sum+target)/2;
vector<vector<int>> res;
vector<int> path;
void traversal(vector<int> &nums, int target, int startidx, int sum){
    if(sum == target){
        res.push_back(path);
    }
    for(int i=startidx;i<nums.size() && sum+nums[i]<=target;i++){
        sum += nums[i];
        path.push_back(nums[i]);
        traversal(nums,target,i+1,sum);
        sum -= nums[i];
        path.pop_back();
    }
}
int mubiaohe(vector<int> &nums, int target){

    int sum=0;
    for(int i=0;i<nums.size();i++)
        sum += nums[i];
    if(target > sum) return 0;
    if((target+sum)%2 == 1) return 0;
    int bag=(sum+target)/2;

    sort(nums.begin(), nums.end());
    traversal(nums, target, 0, 0);
    return res.size();
}

int change(vector<int> &coins,int amount){
    vector<int> dp(coins.size(), 0);
    dp[0]=1;
    for(int i=0;i<coins.size();i++){
        for(int j=coins[i];j<amount;j++){
            dp[j] += dp[j-coins[i]];
        }
    }
    return ddp[amount];    
}

[1,2,5]  5
dp[j]+=dp[j-coins[i]]
           0  1  2  3  4  5
coins[0]   1  1  1  1  1  1
coins[1]   1  1  2  2  3  3
coins[2]   1  1  2  2  3  4

打家劫舍
dp[i] 打劫第i个房屋，最多能偷窃到的金额
vector<int> dp(nums.size(), 0);
dp[0]=nums[0];
dp[1]=max(nums[0], nums[1]);
dp[i]=max(dp[i-1], dp[i-2]+nums[i])

int maxMoney(vector<int> &nums){
    if(nums.size() == 0) return 0;
    if(nums.size() == 1) return nums[0]; 
    vector<int> dp(nums.size());
    dp[0]=nums[0];
    dp[1]=max(nums[0], nums[1]);
    for(int i=2;i<nums.size();i++){
        dp[i]=max(dp[i-1], dp[i-2]+nums[i]);
    }
    return dp[nums.size()-1];
}
打家劫舍2 圈
int rob(vector<int> &nums, int start, int end){
    vector<int> dp(nums.size());
    dp[start]=nums[start];
    dp[start+1]=max(nums[start], nums[start+1]);
    for(int i=start+2;i<=end;i++){
        dp[i]=max(dp[i-1], dp[i-2]+nums[i]);
    }
    return dp[end];
}
int maxMoney2(vector<int> &nums){
    if(nums.size() == 0) return 0;
    if(nums.size() == 1) return nums[0]; 
    int res1 = rob(nums, 0, nums.size()-2);
    int res2 = rob(nums, 1, nums.size()-1);
    return max(res1, res2);
    
}
打家劫舍3 树
int rob3(TreeNode *root){
    if(root==NULL) return 0;
    if(!root->left && !root->right) return root->val;
    //偷父
    int val1=root->val;
    if(root->left) val1 += rob3(root->left->left)+rob3(root->left->right);
    if(root->right) val1 += rob3(root->right->left)+rob3(root->right->right);

    //不偷父
    int val2=rob(root->left)+rob(root->right);
    return max(val1,val2);
}
//买卖股票
int maxprofit(vector<int> &prices){
    int res=0;
    for(int i=0;i<prices.size();i++){
        for(int j=i+1;j<prices.size();j++){
            res = max(res, prices[j]-prices[i]);
        }
    }
    return res;
}

[7,1,5,3,6,4]
dp[][0] -7 -1 -1 -1 -1 -1
dp[][1]  0  0  4  4  5  5

int maxprofit(vector<int> &prices){
    if(prices.size() == 0) return 0;

    vector<vector<int>> dp(prices.size(), vector<int>(2));
    dp[0][0]=-prices[0];
    dp[0][1]=0;
    for(int i=1;i<prices.size();i++){
        dp[i][0]=max(dp[i-1][0], -prices[i]);
        dp[i][1]=max(dp[i-1][1], prices[i]+dp[i-1][0]);
    }
    return dp[prices.size()-1][1];
}
//买卖股票2:多次买卖一支股票
int maxprofit(vector<int> &prices){
    vector<vector<int>> dp(prices.size(), vector<int>(2, 0));
    dp[0][0]=-prices[0];
    dp[0][1]=0;
    for(int i=1;i<prices.size();i++){
        dp[i][0]=max(dp[i-1][0], dp[i-1][1]-prices[i]);
        dp[i][1]=max(dp[i-1][1], dp[i-1][0]+prices[i]);
    }
    return dp[prices.size()-1][1];
}
//买卖股票3:最多买卖2次，再次购买前出售掉之前的股票
0 无操作
1 第一次买入
2 第一次卖出
3 第二次买入
4 第二次卖出
int maxprofit(vector<int> &prices){
    if(prices.size() == 0) return 0;
    vector<int> dp(prices.size(), vector<int>(5, 0));
    dp[0][1]=-prices[0];
    dp[0][3]=-prices[0];

    for(int i=1;i<prices.size();i++){
        dp[i][0]=dp[i-1][0];
        dp[i][1]=max(dp[i-1][1], dp[i-1][0]-prices[i]);
        dp[i][2]=max(dp[i-1][2], dp[i-1][1]+prices[i]);
        dp[i][3]=max(dp[i-1][3], dp[i-1][2]-prices[i]);
        dp[i][4]=max(dp[i-1][4], dp[i-1][3]+prices[i]);
    }
    return dp[prices.size()-1][4];
}

//买卖股票3:最多买卖k次，再次购买前出售掉之前的股票
0 无操作
1 第一次买入
2 第一次卖出
3 第二次买入
4 第二次卖出
int maxprofit(vector<int> &prices){
    if(prices.size() == 0) return 0;
    vector<int> dp(prices.size(), vector<int>(2*k+1, 0));
    for(int j=1;j<2*k;j+=2){
        dp[0][j]=-prices[0];
    }

    for(int i=1;i<prices.size();i++){
        for(int j=0;j<2*k;j+=2){
            dp[i][j+1]=max(dp[i-1][j+1], dp[i-1][j]-prices[i]);
            dp[i][j+2]=max(dp[i-1][j+2], dp[i-1][j+1]+prices[i]);
        }
    }
    return dp[prices.size()-1][2*k];
}
//买卖股票4:多次买卖，冷冻期
0 买入
  卖出
   1之前卖出
   2当天卖出
3  冷冻期
int maxprofit(vector<int> &prices){
    if(prices.size() == 0) return 0;
    vector<int> dp(prices.size(), vector<int>(4, 0));
    dp[0][0]=-prices[0];

    for(int i=0;i<prices.size();i++){
        dp[i][0]=max(dp[i-1][0], max(dp[i-1][3]-prices[i], dp[i-1][1]-prices[i]));
        dp[i][1]=max(dp[i-1][1],dp[i-1][3]);
        dp[i][2]=dp[i-1][0]+prices[i];
        dp[i][3]=dp[i-1][2];

    }
    return max(dp[prices.size()-1][3], max(dp[prices.size()-1][1], prices.size()-1][2]));
}
//买卖股票5:多次买卖，手续费
0 买入
1 卖出
dp[i][0]=max(dp[i-1][0], dp[i-1][1]-prices[i]);
dp[i][1]=max(dp[i-1][1], dp[i-1][0]+prices[i]-fee);
          1   3  2  8  4  9
dp[i][0]  -1 -1  -1 -1 1  1
dp[i][1]  0   0  0  5  5  8
int maxProfix(vector<int> &prices, int fee){
    if(prices.size()==0) return 0;
    vector<int> dp(prices.size(), 0);
    dp[0][0]=-prices[0];
    dp[0][1]=0;
    for(int i=1;i<prices.size();i++){
        dp[i][0]=max(dp[i-1][0], dp[i-1][1]-prices[i]);
        dp[i][1]=max(dp[i-1][1], dp[i-1][0]+prices[i]-fee);
    }
    return max(dp[prices.size()-1][1]);

}
//子序列
//1.最长递增子序列
vector<vector<int>> res;
vector<int> path;
void traversal(vector<int> &nums, int startidx){
    res.push_back(path);
    if(startidx> nums.size()) return;

    unordered_set<int> uset;
    for(int i=startidx;i<nums.size();i++){
        if((!path.empty() &&nums[i]<path.back()) || uset.find(nums[i])!=uset.end()){
            continue;
        }
        uset.insert(nums[i]);
        path.push_back(nums[i]);
        traversal(nums, i+1);
        path.pop_back(nums[i]);
    }
}
int longZiXulie(vector<int> &nums){
    traversal(nums, 0);
    int ret=INT_MIN;
    for(int i=0;i<res.size();i++){
        if(path[i].size() > ret)
            ret=path[i].size();
    }
    return ret;
}

dp[i] j （0 - i-1）
int longZiXulie(vector<int> &nums){
    if(nums.size() == 1) return 1;
    vector<int> dp(nums.size(), 1);
    int res=0;
    for(int i=1;i<nums.size();i++){
        for(int j=0;j<i;j++){
            if(nums[i]>nums[j])
                dp[i]=max(dp[i], dp[j]+1);
        }
         if(dp[i] > res) res=dp[i];
    }
    return res;
}
  0 1 0 3 2
  1 2 1 3 3

int longZiXulie2(vector<int> &nums){
    if(nums.size() <= 1) return nums.size();
    vector<int> dp(nums.size(), 1);
    int res=0;
    for(int i=1;i<nums.size();i++){
        if(nums[i]>nums[i-1])
            dp[i]=dp[i-1]+1;
        if(dp[i] > res) res=dp[i];
    }
    return res;
} 
子数组
int longCopyShuzu(vector<int> &A, vector<int> &B){
    vector<vector<int>> dp(A.size()+1, vector<int>(B.size()+1, 0));
    int res=0;
    for(int i=1;i<=A.size();i++){
        for(int j=1;j<=B.size();j++){
            if(A[i-1]==B[j-1]){
                dp[i][j]=dp[i-1][j-1]+1;
            }
            if(dp[i][j] > res) res=dp[i][j];
        }
    }
    return res;
}
int longCopyShuzu(vector<int> &A, vector<int> &B){
    vector<int> dp(B.size()+1, 0);
    int res=0;
    for(int i=1;i<=A.size();i++){
        for(int j=1;j<=B.size();j++){
            if(A[i-1]==B[j-1]){
                dp[j]=dp[j-1]+1;
            }
            if(dp[j] > res) res=dp[j];
        }
    }
    return res;
}
子序列
int longUnion(string text1, string text2){
    vector<vector<int>> dp(text1.size()+1, vector<int>(text2.size()+1, 0));
    for(int i=1;i<=text1.size();i++){
        for(int j=1;j<=text2.size();j++){
            if(text1[i-1]==text2[j-1]){
                dp[i][j]=dp[i-1][j-1]+1;
            }else{
                dp[i][j]=max(dp[i-1][j], dp[i][j-1]);
            }
        }
        return dp[text1.size()][text2.size()];
    }
}

  [-2,1,-3,4,-1,2,1,-5,4]
   -2 1 -2 4  3 5 6  1  5 
子序和
int maxZixuSum(vector<int> &nums){
    vector<int> dp(nums.size());
    dp[0]=nums[0];
    int res=0;
    for(int i=1;i<nums.size();i++){
        dp[i]=max(dp[i-1]+nums[i], nums[i]);
        if(dp[i] > res) res=dp[i];
    }
    return res;
}

bool isZiXuLie(string s1, string s2){
    vector<vector<int>> dp(s1.size()+1, vector<int>(s2.size()+1, 0));
    for(int i=1;i<s1.size();i++){
        for(int j=1;j<s2.size();j++){
            if(s1[i-1] == s2[j-1]){
                dp[i][j]=dp[i-1][j-1]+1;
            }else{
                dp[i][j]=dp[i][j-1];
            }
        }
    }
    if(dp[s1.size()][s2.size()]==s1.size()) return true;
    else return false;
}

