#include "op_deque.h"
#include <vector>

template<typename T, typename F> int  op_deque<T, F>::size(){
    return s1.size() + s2.size();
}
template<typename T, typename F> T    op_deque<T, F>::operator[](unsigned int i){
    return i < s1.size() ? s1[s1.size()-i-1].first
                         : s2[i-s1.size()].first;
}
template<typename T, typename F> void op_deque<T, F>::balance(){
    int low = std::min(s1.size(), s2.size());
    int high = std::max(s1.size(), s2.size());
    if(3 * low < high){
        int n = this->size(), half = n/2, rest = n-half;
        std::vector<T> af(half), ab(rest);
        for(int i=0; i<half; i++) af[half-i-1] = (*this)[i];
        for(int i=0; i<rest; i++) ab[i]        = (*this)[half+i];
        s1 = rastack<std::pair<T,T>>();
        s2 = rastack<std::pair<T,T>>();
        for(int i=0; i<half; i++){
            T e = i ? op(af[i], s1.top().second) : af[i];
            s1.push({af[i], e});
        }
        for(int i=0; i<rest; i++){
            T e = i ? op(s2.top().second, ab[i]) : ab[i];
            s2.push({ab[i], e});
        }
    }
}
template<typename T, typename F> void op_deque<T, F>::pop_front(){
    if(s1.empty()) s2.pop(); else s1.pop();
    balance();
}
template<typename T, typename F> void op_deque<T, F>::pop_back(){
    if(s2.empty()) s1.pop(); else s2.pop();
    balance();
}
template<typename T, typename F> void op_deque<T, F>::push_front(T x){
    T e = s1.empty() ? x : op(x, s1.top().second);
    s1.push({x, e});
    balance();
}
template<typename T, typename F> void op_deque<T, F>::push_back(T x){
    T e = s2.empty() ? x : op(s2.top().second, x);
    s2.push({x, e});
    balance();
}
template<typename T, typename F> T    op_deque<T, F>::front(){
    return s1.empty() ? s2.top().first
                      : s1.top().first;
}
template<typename T, typename F> T    op_deque<T, F>::back(){
    return s2.empty() ? s1.top().first
                      : s2.top().first;
}
template<typename T, typename F> T    op_deque<T, F>::get(){
    if(s1.empty() or s2.empty())
        return s1.empty() ? s2.top().second
                          : s1.top().second;
    return op(s1.top().second, s2.top().second);
}
