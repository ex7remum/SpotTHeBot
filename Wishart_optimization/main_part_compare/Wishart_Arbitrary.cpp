#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>
#include <queue>
#include <iostream>
#include <cmath>
#include <exception>
#include <unordered_map>
#pragma warning(disable : 4996)

// define dimension of input here
class Point : public std::array<double, 10>
{
public:

    // dimension of the Point
    static const int DIM = 10;

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

    /** @brief Searches the nearest neighbor.
    */
    int nnSearch(const PointT& query, double* minDist = nullptr) const
    {
        int guess;
        double _minDist = std::numeric_limits<double>::max();

        nnSearchRecursive(query, root_, &guess, &_minDist);

        if (minDist)
            *minDist = _minDist;

        return guess;
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

    /** @brief Searches neighbors within radius.
    */
    std::vector<int> radiusSearch(const PointT& query, double radius) const
    {
        std::vector<int> indices;
        radiusSearchRecursive(query, root_, indices, radius);
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

    /** @brief Searches the nearest neighbor recursively.
    */
    void nnSearchRecursive(const PointT& query, const Node* node, int *guess, double *minDist) const
    {
        if (node == nullptr)
            return;

        const PointT& train = points_[node->idx];

        const double dist = distance(query, train);
        if (dist < *minDist)
        {
            *minDist = dist;
            *guess = node->idx;
        }

        const int axis = node->axis;
        const int dir = query[axis] < train[axis] ? 0 : 1;
        nnSearchRecursive(query, node->next[dir], guess, minDist);

        const double diff = fabs(query[axis] - train[axis]);
        if (diff < *minDist)
            nnSearchRecursive(query, node->next[!dir], guess, minDist);
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

    /** @brief Searches neighbors within radius.
    */
    void radiusSearchRecursive(const PointT& query, const Node* node, std::vector<int>& indices, double radius) const
    {
        if (node == nullptr)
            return;

        const PointT& train = points_[node->idx];

        const double dist = distance(query, train);
        if (dist < radius)
            indices.push_back(node->idx);

        const int axis = node->axis;
        const int dir = query[axis] < train[axis] ? 0 : 1;
        radiusSearchRecursive(query, node->next[dir], indices, radius);

        const double diff = fabs(query[axis] - train[axis]);
        if (diff < radius)
            radiusSearchRecursive(query, node->next[!dir], indices, radius);
    }

    Node* root_;                 //!< root node
    std::vector<PointT> points_; //!< points
};

#include <chrono>
class Wishart {
public:
    Wishart(int neighbours_, double significance_level_) :
        neighbours(neighbours_), significance_level(significance_level_) {
        clusters = { { 1, 1, 0 } };
        mult_const = 1.0 * neighbours_ / pow(acos(-1.0), Point::DIM / 2) * tgamma(1.0 * Point::DIM / 2.0 + 1);
    };

    void fit(const std::vector<Point>& points) {
        int size = points.size();
        double significance_const = mult_const / (1.0 * points.size());
        object_labels.clear();
        object_labels.resize(size, -1);
        clusters_to_objects.clear();
        clusters_to_objects.resize(size);
        using std::chrono::high_resolution_clock;
        using std::chrono::duration_cast;
        using std::chrono::duration;
        using std::chrono::milliseconds;

        auto t1 = high_resolution_clock::now();

        KDTree<Point> tree(points);
        std::vector<std::vector<int>> indexes_of_k_neighbours;
        for (size_t i = 0; i < points.size(); ++i) {
            std::vector<int> query_result = tree.knnSearch(points[i], neighbours + 1); //because you are your own neighbour
            indexes_of_k_neighbours.push_back({});
            for (size_t i = 1; i < query_result.size(); ++i) {
                indexes_of_k_neighbours.back().push_back(query_result[i]);
            }
        }

        auto t2 = high_resolution_clock::now();
        auto ms_int = duration_cast<milliseconds>(t2 - t1);

        duration<double, std::milli> ms_double = t2 - t1;
        std::cout << "KDTree spent time: " << ms_double.count() << "ms\n";
        std::vector<double> distances;

        for (size_t i = 0; i < points.size(); ++i) {
            Point last = points[indexes_of_k_neighbours[i].back()];
            double dist = 0;
            for (size_t j = 0; j < Point::DIM; ++j) {
                dist += (points[i][j] - last[j]) * (points[i][j] - last[j]);
            }
            distances.push_back(sqrt(dist));
        }
        std::vector<int> indexes(size);
        iota(indexes.begin(), indexes.end(), 0);
        std::sort(indexes.begin(), indexes.end(), [&](int a, int b) {
            return distances[a] < distances[b];
        });

        for (int index : indexes) {
            std::vector<int> neighbours_clusters;
            for (int j : indexes_of_k_neighbours[index]) {
                neighbours_clusters.push_back(object_labels[j]);
            }
            std::sort(neighbours_clusters.begin(), neighbours_clusters.end());
            neighbours_clusters.resize(
                std::unique(neighbours_clusters.begin(), neighbours_clusters.end()) - neighbours_clusters.begin());
            std::vector<int> unique_clusters;
            for (int j : neighbours_clusters) {
                if (j != -1) {
                    unique_clusters.push_back(j);
                }
            }

            if (unique_clusters.empty()) {
                create_new_cluster(index, distances[index]);
            }
            else {
                int min_cluster = unique_clusters[0];
                int max_cluster = unique_clusters.back();
                if (min_cluster == max_cluster) {
                    if (clusters[max_cluster].back() < 0.5) {
                        add_elem_to_exist_cluster(index, distances[index], max_cluster);
                    }
                    else {
                        add_elem_to_noise(index);
                    }
                }
                else {
                    std::vector<std::vector<double>> my_clusters;
                    std::vector<double> flags;
                    for (int j : unique_clusters) {
                        my_clusters.push_back(clusters[j]);
                        flags.push_back(clusters[j].back());
                    }
                    double flags_min = flags[0];
                    for (double val : flags) {
                        flags_min = std::min(val, flags_min);
                    }
                    if (flags_min > 0.5) {
                        add_elem_to_noise(index);
                    }
                    else {
                        std::vector<double> significan;
                        for (const auto& cluster : my_clusters) {
                            // needs to be adaptated to bigger dimensions
                            double cur_significan = bin_pow(1.0 / cluster[0], Point::DIM) - bin_pow(1.0 / cluster[1], Point::DIM);
                            cur_significan *= significance_const;
                            significan.push_back(cur_significan);
                        }
                        std::vector<int> significan_clusters, not_significan_clusters;
                        for (size_t j = 0; j < significan.size(); ++j) {
                            if (significan[j] >= significance_level) {
                                significan_clusters.push_back(unique_clusters[j]);
                            }
                            else {
                                not_significan_clusters.push_back(unique_clusters[j]);
                            }
                        }
                        int significan_cluster_count = significan_clusters.size();
                        if (significan_cluster_count > 1 || min_cluster == 0) {
                            add_elem_to_noise(index);
                            for (int significan_cluster : significan_clusters) {
                                clusters[significan_cluster].back() = 1;
                            }
                            for (int not_sig_cluster : not_significan_clusters) {
                                if (not_sig_cluster == 0) {
                                    continue;
                                }
                                for (int bad_index : clusters_to_objects[not_sig_cluster]) {
                                    add_elem_to_noise(bad_index);
                                }
                                clusters_to_objects[not_sig_cluster].clear();
                            }
                        }
                        else {
                            for (int cur_cluster : unique_clusters) {
                                if (cur_cluster == min_cluster) {
                                    continue;
                                }
                                for (int bad_index : clusters_to_objects[cur_cluster]) {
                                    add_elem_to_exist_cluster(bad_index, distances[bad_index], min_cluster);
                                }
                                clusters_to_objects[cur_cluster].clear();
                            }
                            add_elem_to_exist_cluster(index, distances[index], min_cluster);
                        }
                    }
                }
            }
        }
        clean_data();
        //for (auto i : object_labels) {
        //    std::cout << i << ", ";
        //}
        //std::cout << "\n";
    }

private:
    int neighbours;
    double significance_level;
    double mult_const;
    std::vector<std::vector<double>> clusters;
    std::vector<int> object_labels;
    std::vector<std::vector<int>> clusters_to_objects;

    void clean_data() {
        int unique_cluster_counter = 1;
        int ind = 0;
        std::unordered_map<int, int> true_labels;
        std::vector<int> result(object_labels.size());
        true_labels[0] = 0;
        for (int label : object_labels) {
            if (label == 0) {
                result[ind] = 0;
            } else {
                if (true_labels.find(label) == true_labels.end()) {
                    true_labels[label] = unique_cluster_counter;
                    ++unique_cluster_counter;
                }
                result[ind] = true_labels[label];
            }
            ++ind;
        }
        std::cout << unique_cluster_counter - 1 << "\n";
        object_labels = result;
    }

    inline void create_new_cluster(int index, double dist) {
        object_labels[index] = clusters.size();
        clusters_to_objects[clusters.size()].push_back(index);
        clusters.push_back({ dist, dist, 0 });
    }

    inline void add_elem_to_noise(int index) {
        object_labels[index] = 0;
        clusters_to_objects[0].push_back(index);
    }

    inline void add_elem_to_exist_cluster(int index, double dist, int cluster_label) {
        object_labels[index] = cluster_label;
        clusters_to_objects[cluster_label].push_back(index);
        clusters[cluster_label][0] = std::min(clusters[cluster_label][0], dist);
        clusters[cluster_label][1] = std::max(clusters[cluster_label][1], dist);
    }

    inline double bin_pow(double val, int p) {
        double res = 1.0;
        while (p > 0) {
            if (p & 1) {
                res *= val;
            }
            val *= val;
            p /= 2;
        }
        return res;
    }
};


#include <chrono>
int main(int argc, char **argv) {
    if (argc < 3) {
        std::cout << "Too low arguments\n";
        return 0;
    }
    std::cout << "Enter parameters of Wishart algorithm\n";
    fflush(stdout);
    int neighbours;
    double significance;
    std::cin >> neighbours >> significance;
    FILE* fin = freopen(argv[1], "r", stdin);
    FILE* fout = freopen(argv[2], "w", stdout);
    std::vector<Point> pts;
    std::vector<double> cur_point(Point::DIM);
    int i = 0;
    while (true) {
        int need = 0;
        while (need < Point::DIM && std::cin >> cur_point[need]) {
            ++need;
        }
        if (need != Point::DIM) {
            break;
        }
        Point pt(cur_point, i);
        pts.push_back(pt);
        ++i;
    }
    //fclose(fin);
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    Wishart W(neighbours, significance);
    auto t1 = high_resolution_clock::now();
    W.fit(pts);
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

    duration<double, std::milli> ms_double = t2 - t1;
    //fclose(fout);
    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n";
}

