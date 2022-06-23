

### 一、基础数据结构

|类型|场景|使用方法|
|-|-|-|
|array|适应于大量查、改操作，不适应于大量交换、删除、增加操作|armyarrayray[i] = i, myarray.size(), auto it=myarray.begin(); it != myarray.end(); it++, cout << *it << endl|
|vector|可以根据数据大小自动调整，这里包含初始化、插入、删除等操作|myvector.push_back(i);其余和上面一致|
|list链表|list链表：内存上为链式结构，可以动态增加，适用于大量增加，删除的应用，不适合大量查询的应用|mylist.insert(it, i);遍历操作和上面一样|
| queue队列|先进先出FIFO，动态调整大小，适用于大量增、删操作，不适应于大量查找操作|myqueue.push(i), myqueue.empty(), myqueue.front(), myqueue.pop()|
| stack栈|先进后出，适应于图的遍历、递归函数的改写等|mystack.push(),mystack.top(),mystack.pop(),mystack.empty()|
|map|kv|map<char,int>, mymap;search = mymap.find('a'), search != mymap.end(), search->first, search->second();|
|set|set|set<int>, myset.insert(i),遍历和前面一样, myset.find(i) != myset.end(),myset.erase (myset.find(40));|
|deque|双端队列|mydeque.push_back(), mydeque.push_front(),mydeque.pop_front(), mydeque.pop_back()|
|priority_queue|最大堆,最小堆|最小堆：std::priority_queue<int, std::vector<int>, std::greater<int>>, 最大堆(默认)：std::priority_queue<int>, 最大堆: priority_queue<int, std::vector<int>, std::less<int> maxqueue, maxqueue.top(), maxqueue.top()|

#### 1.1 array


适应于大量查、改操作，不适应于大量交换、删除、增加操作


```c++
#import <iostream>
#import <string>
#import <array>
using namespace std;
int main() 
{
  array<int, 10> myarray;
  for(int i=0; i<10; i++) {
    myarray[i] = i;
  }
  for(auto it = myarray.begin(); it != myarray.end(); it++) {
    cout << *it << '\t';
  }
  cout << endl;
  cout << "size:" << myarray.size() << endl;
  cout << "myarray[4]:" << myarray[4] << endl;
	return 0;
}

```
- output
```
0 1 2 3 4 5 6 7 8 9
size:10
myarray[4]:4
```


#### 1.2 vector


```c++
#import <iostream>
#import <string>
#import <vector>

using namespace std;

//可以根据数据大小自动调整，这里包含初始化、插入、删除等操作

int main() 
{
  vector<int> myvector;
  for(int i=0; i<10; i++) {
    myvector.push_back(i);
  }
  
  for(auto it = myvector.begin(); it != myvector.end(); it++) {
    cout << *it << '\t';
  }
  
  cout << endl;
  cout << "size:" << myvector.size() << endl;
  cout << "myvector[4]:" << myvector[4] << endl;
	return 0;
}

// 输出和上面一致
```

#### 1.3 list链表

```c++
#import <iostream>
#import <string>
#import <list>

using namespace std;

// list链表：内存上为链式结构，可以动态增加，适用于大量增加，删除的应用，不适合大量查询的应用

int main() 
{
  
  int num[] = {1,2,3,4,5};
  list<int> mylist(num, num+sizeof(num)/sizeof(int));
  
  for(auto it = mylist.begin(); it != mylist.end(); it++) {
    cout << *it << '\t';
  }
  
  auto it = mylist.begin();
  for(int i=0; i<5; i++) {
    mylist.insert(it, i);
  }
  
  cout << endl;
  
    for(auto it = mylist.begin(); it != mylist.end(); it++) {
    cout << *it << '\t';
  }

	return 0;
}

```

- output

```
1 2 3 4 5
0 1 2 3 4 1 2 3 4 5
```


#### 1.4 queue FIFO队列

```c++
#import <iostream>
#import <string>
#import <queue>

using namespace std;

// queue队列：先进先出FIFO，动态调整大小，适用于大量增、删操作，不适应于大量查找操作

int main() 
{
  queue<int> myqueue;
  for(int i=0; i<10; i++) {
    myqueue.push(i);
  }
  while(!myqueue.empty()) {
    // FIFO
    cout << myqueue.front() << '\t';
    myqueue.pop();
  }
	return 0;
}

```
- output

```
0 1 2 3 4 5 6 7 8 9
```


#### 1.4 stack 栈

```c++

#import <iostream>
#import <string>
#import <stack>

using namespace std;
// stack栈：先进后出，适应于图的遍历、递归函数的改写等
int main() 
{
  stack<int> mystack;
  for(int i=0; i<10; i++) {
    mystack.push(i);
  }
  while(!mystack.empty()) {
    cout << mystack.top() << '\t';
    mystack.pop();
  }
	return 0;
}
```
-output
```
9 8 7 6 5 4 3 2 1 0
```

#### 1.5 map

```c++
#import <iostream>
#import <string>
#import <map>

using namespace std;

// map：kv

int main() 
{
  map<char,int> mymap;
  
  mymap['a'] = 1;
  mymap['b'] = 2;
  mymap.erase('a');
  auto search = mymap.find('b');
  if (search != mymap.end()) {
        std::cout << "Found " << search->first << " " << search->second << '\n';
    } else {
        std::cout << "Not found\n";
    }
   search = mymap.find('a');
   if (search != mymap.end()) {
        std::cout << "Found " << search->first << " " << search->second << '\n';
    } else {
        std::cout << "Not found\n";
    }
}


```


#### 1.6 set

```c++
#include <iostream>
#include <set>

using namespace std;
int main ()
{
  std::set<int> myset;
  std::set<int>::iterator it;

  // set some initial values:
  for (int i=1; i<=5; i++) myset.insert(i*10);    // 
  for (it=myset.begin(); it!=myset.end(); ++it)
    std::cout << ' ' << *it;
  
  cout << std::endl;
  
  it = myset.find(20);
  myset.erase (it);
  myset.erase (myset.find(40));

  std::cout << "myset contains:";
  for (it=myset.begin(); it!=myset.end(); ++it)
    std::cout << ' ' << *it;
  std::cout << '\n';
  
  if(myset.find(60) != myset.end()) {
    cout << "find:60" << endl;
  } else {
    cout << "not find:60" << endl;
  }
  return 0;
}
```

- output

```
10 20 30 40 50
myset contains: 10 30 50
not find:60
```

#### 1.6 deque 双端队列

```c++
#include <iostream>
#include <deque>
 
int main()
{
    // Create a deque containing integers
    std::deque<int> d = {7, 5, 16, 8};
    // Add an integer to the beginning and end of the deque
    d.push_front(13);
    d.push_back(25);
    // Iterate and print values of deque
    for(int n : d) {
        std::cout << n << ' ';
    }
  std::cout << std::endl;
    std::cout << "d.front():" << d.front() << std::endl;
    std::cout << "d.back():" << d.back() << std::endl;
    d.pop_front();
   std::cout << "d.front() after pop_front:" << d.front() << std::endl;
    d.pop_back();
    std::cout << "d.back() after pop_back:" << d.back() << std::endl;
    
}
```

- output
```
3 7 5 16 8 25
d.front():13
d.back():25
d.front() after pop_front:7
d.back() after pop_back:8
```

#### 1.7 priority_queue 最大堆最小堆

```c++
#include <iostream>
#include <queue>
 
template<typename T>
void print_queue(T q) { // NB: pass by value so the print uses a copy
    while(!q.empty()) {
        std::cout << q.top() << ' ';
        q.pop();
    }
    std::cout << '\n';
}

int main()
{
  // 默认是最大堆
  std::priority_queue<int> q;
  //等同于 priority_queue<int, vector<int>, less<int> > a;
  q.push(5);
  q.push(3);
  q.push(8);
  q.push(2);
  q.push(100);
  
  print_queue(q);
    
  // 最小堆
  std::priority_queue<int, std::vector<int>, std::greater<int>> q2;
  q2.push(5);
  q2.push(3);
  q2.push(8);
  q2.push(2);
  q2.push(100);
  print_queue(q2);
  
}
```

- output
```
100 8 5 3 2
2 3 5 8 100
```
