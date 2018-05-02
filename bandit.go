package main

import ()

func (u *User) ContextVector() []float64 {

}

type BanditItem struct {
	TimesPlayed int
	MeanOutcome float64
}

type BanditArm struct {
}

func GenItems(u User) Items {
	return Items{}
}
