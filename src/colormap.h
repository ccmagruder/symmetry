#pragma once

class color
{
    public:
        color(uint16_t r=0, uint16_t g=0, uint16_t b=0){
            _ptr = new uint16_t[3];
            _r = _ptr;
            _g = _ptr + 1;
            _b = _ptr + 2;
            *_r = r;
            *_g = g;
            *_b = b;
        }

        // Shallow copy to ptr
        explicit color(uint16_t* r):_r(r),_g(r+1),_b(r+2) {}
        // Dealloc not needed, ptr=NULL

        // Shallow copy constructor
        color(const color &c) : color(c._r) {}
        // Dealloc not needed, ptr=NULL

        color& operator=(const color& c){
            _ptr = new uint16_t[3];
            _r = _ptr;
            _g = _ptr+1;
            _b = _ptr+2;
            *_r = *(c._r);
            *_g = *(c._g);
            *_b = *(c._b);
            return *this;
        }

        ~color(){ delete _ptr; _ptr = NULL; }
        // Does nothing when _ptr==NULL from shallow copy

        uint16_t* _r = NULL;
        uint16_t* _g = NULL;
        uint16_t* _b = NULL;
        uint16_t* _ptr = NULL; // Ptr to remember whether allocated
    
};

class colormap
{
    public:
        colormap(char* fileName){
            // Count number of lines
            _numColors = 0;
            std::string line;
            std::ifstream file;
            file.open(fileName);
            while (std::getline(file, line))
                ++_numColors;
            file.clear(); // Reset ifstream eof and fail flags
            file.seekg(0); // Reset ifstream buffer

            _alpha = new uint16_t[_numColors];
            _map = new color[_numColors];
            for (ptrdiff_t i=0; i<_numColors; i++){
                file >> _alpha[i];
                file >> *(_map[i]._r);
                file >> *(_map[i]._g);
                file >> *(_map[i]._b);

                // Rescale RGB[0,255] -> RGB[0, 65535]
                // __UINT16_MAX__ / __UINT8_MAX__ == __UINT8_MAX__ + 2 = 257
                *(_map[i]._r) = uint16_t(*(_map[i]._r) * ( __UINT8_MAX__ + 2 ));
                *(_map[i]._g) = uint16_t(*(_map[i]._g) * ( __UINT8_MAX__ + 2 ));
                *(_map[i]._b) = uint16_t(*(_map[i]._b) * ( __UINT8_MAX__ + 2 ));
            }
            file.close();

        };

        // Shallow copy constructor
        colormap(const colormap &map) : _alpha(map._alpha), _map(map._map), _numColors(map._numColors) {}

        color lookup(uint16_t grey);
        uint16_t* _alpha;
        color* _map;
        size_t _numColors;
};

color interpolate(const color& a, const color& b, const uint16_t& alpha);
void map2monochrome(image *imG, image* imRGB, const color& fill);
void map2multichrome(image *imG, image* imRGB, colormap& map);
void remap(image* imG);
