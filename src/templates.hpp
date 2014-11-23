#pragma once
template <class T, class BaseIterator>
class filter {
    BaseIterator _begin, _end;
    std::function<T> f;
public:
    struct filter_iterator : BaseIterator {
        filter_iterator(BaseIterator base): BaseIterator(base) {}
        typename std::function<T>::result_type operator*() {
            return f(BaseIterator::operator*());
        } 
        
    };
    filter(BaseIterator begin, BaseIterator end, std::function<T>f):
        _begin(begin), _end(end), f(f){}
    filter_iterator begin(){return _begin;}
    filter_iterator end() {return _end;}
};
