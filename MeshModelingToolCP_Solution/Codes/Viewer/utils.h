#pragma once
#include <glm.hpp>
#include <random>


class Random
{
public:
	// Return a random float [min. max]
	static float GetRandom(float min, float max)
	{
		static std::default_random_engine pseudo_random_generator;
		static std::uniform_real_distribution<float> float_distribution;

		
		float_distribution = std::uniform_real_distribution<float>(min, max);
		return float_distribution(pseudo_random_generator);
	}

	Random(Random const&) = delete;
	void operator=(Random const&) = delete;
private:
	Random() {}
};

