#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>
#include <queue>
#include <iostream>
#include <cmath>
#include <exception>
#include <unordered_map>
#include <chrono>
// define dimension of input herclass Point : public std::array<double, 10>
class Point : public std::array<double, 15>
{
public:

    // dimension of the Point
    static const int DIM = 15;

    Point() {};
    Point(std::vector<double> coords_, int index_) {
        index = index_;
        for (int i = 0; i < DIM; ++i) {
            (*this)[i] = coords_[i];
        }
    }

private:
    int index;
};

template <class PointT>
class KDTree
{
public:
    /** @brief The constructors.
    */
    KDTree() : root_(nullptr) {};
    KDTree(const std::vector<PointT>& points) : root_(nullptr) { build(points); }

    /** @brief The destructor.
    */
    ~KDTree() { clear(); }

    /** @brief Re-builds k-d tree.
    */
    void build(const std::vector<PointT>& points)
    {
        clear();

        points_ = points;

        std::vector<int> indices(points.size());
        std::iota(std::begin(indices), std::end(indices), 0);

        root_ = buildRecursive(indices.data(), (int)points.size(), 0);
    }

    /** @brief Clears k-d tree.
    */
    void clear()
    {
        clearRecursive(root_);
        root_ = nullptr;
        points_.clear();
    }

    /** @brief Validates k-d tree.
    */
    bool validate() const
    {
        try
        {
            validateRecursive(root_, 0);
        }
        catch (const Exception&)
        {
            return false;
        }

        return true;
    }

    /** @brief Searches k-nearest neighbors.
    */
    std::vector<int> knnSearch(const PointT& query, int k) const
    {
        KnnQueue queue(k);
        knnSearchRecursive(query, root_, queue, k);

        std::vector<int> indices(queue.size());
        for (size_t i = 0; i < queue.size(); i++)
            indices[i] = queue[i].second;

        return indices;
    }

private:

    /** @brief k-d tree node.
    */
    struct Node
    {
        int idx;       //!< index to the original point
        Node* next[2]; //!< pointers to the child nodes
        int axis;      //!< dimension's axis

        Node() : idx(-1), axis(-1) { next[0] = next[1] = nullptr; }
    };

    /** @brief k-d tree exception.
    */
    class Exception : public std::exception { using std::exception::exception; };

    /** @brief Bounded priority queue.
    */
    template <class T, class Compare = std::less<T>>
    class BoundedPriorityQueue
    {
    public:

        BoundedPriorityQueue() = delete;
        BoundedPriorityQueue(size_t bound) : bound_(bound) { elements_.reserve(bound + 1); };

        void push(const T& val)
        {
            auto it = std::find_if(std::begin(elements_), std::end(elements_),
                [&](const T& element) { return Compare()(val, element); });
            elements_.insert(it, val);

            if (elements_.size() > bound_)
                elements_.resize(bound_);
        }

        const T& back() const { return elements_.back(); };
        const T& operator[](size_t index) const { return elements_[index]; }
        size_t size() const { return elements_.size(); }

    private:
        size_t bound_;
        std::vector<T> elements_;
    };

    /** @brief Priority queue of <distance, index> pair.
    */
    using KnnQueue = BoundedPriorityQueue<std::pair<double, int>>;

    /** @brief Builds k-d tree recursively.
    */
    Node* buildRecursive(int* indices, int npoints, int depth)
    {
        if (npoints <= 0)
            return nullptr;

        const int axis = depth % PointT::DIM;
        const int mid = (npoints - 1) / 2;

        std::nth_element(indices, indices + mid, indices + npoints, [&](int lhs, int rhs)
        {
            return points_[lhs][axis] < points_[rhs][axis];
        });

        Node* node = new Node();
        node->idx = indices[mid];
        node->axis = axis;

        node->next[0] = buildRecursive(indices, mid, depth + 1);
        node->next[1] = buildRecursive(indices + mid + 1, npoints - mid - 1, depth + 1);

        return node;
    }

    /** @brief Clears k-d tree recursively.
    */
    void clearRecursive(Node* node)
    {
        if (node == nullptr)
            return;

        if (node->next[0])
            clearRecursive(node->next[0]);

        if (node->next[1])
            clearRecursive(node->next[1]);

        delete node;
    }

    /** @brief Validates k-d tree recursively.
    */
    void validateRecursive(const Node* node, int depth) const
    {
        if (node == nullptr)
            return;

        const int axis = node->axis;
        const Node* node0 = node->next[0];
        const Node* node1 = node->next[1];

        if (node0 && node1)
        {
            if (points_[node->idx][axis] < points_[node0->idx][axis])
                throw Exception();

            if (points_[node->idx][axis] > points_[node1->idx][axis])
                throw Exception();
        }

        if (node0)
            validateRecursive(node0, depth + 1);

        if (node1)
            validateRecursive(node1, depth + 1);
    }

    static double distance(const PointT& p, const PointT& q)
    {
        double dist = 0;
        for (size_t i = 0; i < PointT::DIM; i++)
            dist += (p[i] - q[i]) * (p[i] - q[i]);
        return sqrt(dist);
    }

    /** @brief Searches k-nearest neighbors recursively.
    */
    void knnSearchRecursive(const PointT& query, const Node* node, KnnQueue& queue, int k) const
    {
        if (node == nullptr)
            return;

        const PointT& train = points_[node->idx];

        const double dist = distance(query, train);
        queue.push(std::make_pair(dist, node->idx));

        const int axis = node->axis;
        const int dir = query[axis] < train[axis] ? 0 : 1;
        knnSearchRecursive(query, node->next[dir], queue, k);

        const double diff = fabs(query[axis] - train[axis]);
        if ((int)queue.size() < k || diff < queue.back().first)
            knnSearchRecursive(query, node->next[!dir], queue, k);
    }

    Node* root_;                 //!< root node
    std::vector<PointT> points_; //!< points
};


static double distance(const Point& p, const Point& q)
{
    double dist = 0;
    for (size_t i = 0; i < Point::DIM; i++)
        dist += (p[i] - q[i]) * (p[i] - q[i]);
    return sqrt(dist);
}

// Dimension number should be defined inside the code in Point class
int main(int argc, char **argv) {
    // input output number_of_points number_of_neighbours
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    if (argc < 5) {
        std::cout << "Too low arguments.\n";
        return 0;
    }
    FILE* fin = freopen(argv[1], "r", stdin);
    FILE* fout = freopen(argv[2], "w", stdout);
    int cnt = atoi(argv[3]);
    int neighbours = atoi(argv[4]);
    std::vector<Point> points;
    for (int i = 0; i < cnt; ++i) {
        std::vector<double> cur_point(Point::DIM);
        for (int j = 0; j < Point::DIM; ++j) {
            std::cin >> cur_point[j];
        }
        Point pt(cur_point, i);
        points.push_back(pt);
    }
    auto t1 = high_resolution_clock::now();
    KDTree<Point> tree(points);
    std::vector<std::vector<int>> indexes_of_k_neighbours;
    std::vector<double> dists;
    for (size_t i = 0; i < points.size(); ++i) {
        std::vector<int> query_result = tree.knnSearch(points[i], neighbours + 1); 
        indexes_of_k_neighbours.push_back({});
        for (size_t i = 1; i < query_result.size(); ++i) {
            indexes_of_k_neighbours.back().push_back(query_result[i]);
        }
        int pos = indexes_of_k_neighbours.back().back();
        double dist = 0;
        for (size_t j = 0; j < Point::DIM; j++)
            dist += (points[i][j] - points[pos][j]) * (points[i][j] - points[pos][j]);
        dists.push_back(sqrt(dist));
    }
    for (auto i : indexes_of_k_neighbours) {
        for (auto j : i) {
            std::cout << j << " ";
        }
        std::cout << "\n";
    }
    for (auto i : dists) {
        std::cout << i << " ";
    }
    std::cout << "\n";
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms\n";
}


