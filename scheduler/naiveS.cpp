#include<bits/stdc++.h>
#include<unistd.h>
using namespace std;
map<int , int> GetVNode;
set<int> collect, newItem, removeItem;
vector<int> numMission, startVCPU, endVCPU;
int mysleep(){
    // usleep(100);
    sleep(1);
    return 0;
}
int getCPUinfo(int nodeNum, int vcPerNode){
    for (int i=0;i<nodeNum;++i){
        numMission.push_back(0);
        startVCPU.push_back(vcPerNode*i);
        endVCPU.push_back(vcPerNode*i+vcPerNode-1);
    }
    return 0;
}
int findSpareNode(){
    int minMission = 99999;
    for (int i=0;i<numMission.size();++i){
        minMission = min(numMission[i],minMission);
    }
    for (int i=0;i<numMission.size();++i){
        if (numMission[i]==minMission) return i;
    }
    // assert(false);
    return 0;
}
int tasksetter(int pid, int cpufrom, int cputo){
    char buffer[200];
    sprintf(buffer, "taskset -a -cp %d-%d %d",cpufrom,cputo,pid);
    system(buffer);
    mysleep();
    return 0;
}
int getPidFromExample(){
    system("ps ax | grep \"python \"| grep -v grep > load.txt");
    mysleep();
    FILE *f = fopen("load.txt","rb");
    int pid = -1;
    collect.clear();
    // pre.clear();
    newItem.clear();
    removeItem.clear();
    while (fscanf(f, "%d", &pid)!=EOF){
        collect.insert(pid);
        // printf("%d\n",pid);
        int c=0;
        while ((c=fgetc(f))!=EOF){
            if (c=='\n'){
                break;
            }
        }
        if (c==EOF) break;
    }
    for (map<int,int>::iterator it=GetVNode.begin();it!=GetVNode.end();++it){
        if (collect.count(it->first)==0){
            removeItem.insert(it->first);
        }
        // fprintf(stdout,"%d -> %d\n",it->first,it->second);
    }
    // fprintf(stdout,"\n");
    for (set<int>::iterator it=collect.begin();it!=collect.end();++it){
        if (GetVNode.count(*it)==0){
            newItem.insert(*it);
        }
    }
    return 0;
}
int sched(){
    getPidFromExample();
    for (set<int>::iterator it = removeItem.begin();it!=removeItem.end();++it){
        --numMission[GetVNode[*it]];
        GetVNode.erase(*it);
    }
    for (set<int>::iterator it = newItem.begin();it!=newItem.end();++it){
        int node = findSpareNode();
        GetVNode[*it] = node;
        ++numMission[node];
        tasksetter(*it, startVCPU[node], endVCPU[node]);
    }
    return 0;
}
int main(){
    {
        FILE * f = fopen("cpuinfo.txt","rb");
        int CpuPerNode, NumOfNode;
        fscanf(f,"%d%d",&CpuPerNode,&NumOfNode);
        getCPUinfo(NumOfNode,CpuPerNode);
    }
    while (true){
        sched();
        // sleep(1);
        mysleep();
    }
}