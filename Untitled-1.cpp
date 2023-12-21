#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main()
{   
    int a;
    a = 500;
    cout<<a<<endl;
    
    int * p;
    p = &a;
    cout<<p<<endl;
    cout<<*p<<endl;
    delete p;
    return 0;
}