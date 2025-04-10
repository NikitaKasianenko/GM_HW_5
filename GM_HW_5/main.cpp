#include <immintrin.h>
#include "iostream"
struct vector4 {
private:
	__m128 data;
public:

	vector4(float x, float y, float z) {
		data = _mm_set_ps(0.0f, z, y, x);
	}

	vector4(float x, float y, float z,float w) {
		data = _mm_set_ps(w, z, y, x);
	}

	vector4& add(const vector4& other) {
		data = _mm_add_ps(data, other.data);
		return *this;
	}

	vector4& add(float x, float y, float z) {
		__m128 temp = _mm_set_ps(0.0f, z, y, x);
		data = _mm_add_ps(data, temp);
		return *this;
	}

	vector4& sub(const vector4& other) {
		data = _mm_sub_ps(data, other.data);
		return *this;
	}

	vector4& sub(float x, float y, float z) {
		__m128 temp = _mm_set_ps(0.0f, z, y, x);
		data = _mm_sub_ps(data, temp);
		return *this;
	}

	vector4& mul(float scale) {
		__m128 scalar = _mm_set_ps1(scale);
		data = _mm_mul_ps(data, scalar);
		return *this;
	}

	vector4& mul(float scale,float w_scale) {
		__m128 scalar = _mm_set_ps(w_scale,scale,scale,scale);
		data = _mm_mul_ps(data, scalar);
		return *this;
	}

	vector4& div(float scale) {
		__m128 scalar = _mm_set_ps1(scale);
		data = _mm_div_ps(data, scalar);
		return *this;
	}

	vector4& div(float scale, float w_scale) {
		__m128 scalar = _mm_set_ps(w_scale, scale, scale, scale);
		data = _mm_div_ps(data, scalar);
		return *this;
	}

	float dot(const vector4& other) const {
		return _mm_cvtss_f32(_mm_dp_ps(data, other.data, 0b11110001));
	}

	float dot(float x,float y, float z) const {
		__m128 temp = _mm_set_ps(0.0f, z, y, x);
		return _mm_cvtss_f32(_mm_dp_ps(data, temp, 0b11110001));
	}

	float magnitude() const {
		return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(dot(*this))));
	}

	float magnitude_square() const {
		return dot(*this);
	}

	vector4& normalize() {
		float mag = magnitude();
		if (mag > 0.0f) {
			data = _mm_div_ps(data, _mm_set1_ps(mag));
		}
		return *this;
	}



};


