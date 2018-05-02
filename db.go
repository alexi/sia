package main

import (
	"encoding/json"
	"io/ioutil"
)

var (
	ItemsList []Item
	ItemsMap  map[string]Item
)

func LoadItems() {
	file, err := ioutil.ReadFile("items.json")
	if err != nil {
		panic(err)
	}
	err = json.Unmarshal(file, &ItemsList)
	if err != nil {
		panic(err)
	}
	for _, item := range ItemsList {
		ItemsMap[item.Id] = item
	}
}

func init() {
	ItemsList = make([]Item, 0)
	ItemsMap = make(map[string]Item)
	LoadItems()

}

// Save DB data to json file
func saveDB() {

}
