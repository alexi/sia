package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"html/template"
	"net/http"
	"net/url"
	"strings"
)

type User struct {
	Age    int    `json:"age"`
	Gender string `json:"gender"`
}

type Item struct {
	Id       string `json:"id"`
	Category string `json:"category"`
}

func (i *Item) getHtml() string {
	return fmt.Sprintf("<div class=\"item\" data-item-id=\"%s\" data-item-category=\"%s\">%s</div>\n", i.Id, i.Category, i.Category)
}

type Items []Item

func (i Items) getHtml() string {
	var str string
	for _, item := range i {
		str += item.getHtml()
	}
	return str
}

type HTMLBody struct {
	BodyString string
}

func parseUser(userJson string) User {
	var u User
	if err := json.Unmarshal([]byte(userJson), &u); err != nil {
		panic(err)
	}
	return u
}

func loadPage(w http.ResponseWriter, r *http.Request, user User) {
	items := GenItems(user)
	itemsHtml := items.getHtml()
	buf := new(bytes.Buffer)
	t, _ := template.ParseFiles("page.html")
	err := t.Execute(buf, HTMLBody{itemsHtml})
	if err != nil {
		panic(err)
	}
	w.WriteHeader(200)
	fmt.Fprintf(w, buf.String())
}

func handleClick(w http.ResponseWriter, r *http.Request, user User) {
	w.WriteHeader(200)
}

func handler(w http.ResponseWriter, r *http.Request) {
	defer func() {
		if r := recover(); r != nil {
			errmsg := fmt.Sprint("Error:", r)
			fmt.Println(errmsg)
			w.WriteHeader(500)
			w.Write([]byte(errmsg))
		}
	}()
	fmt.Println("handling request")
	fmt.Println(r.Cookies())
	cookie, err := r.Cookie("user")
	if err != nil {
		fmt.Println(err)
		panic(err)
	}

	jsonStr, err := url.QueryUnescape(strings.Split(cookie.String(), "=")[1])
	if err != nil {
		panic(err)
	}
	u := parseUser(jsonStr)

	req := r.URL.Path
	req = strings.TrimPrefix(req, "/")
	switch req {
	case "click":
		handleClick(w, r, u)
	default:
		loadPage(w, r, u)
	}
}

func main() {
	defer func() {
		if r := recover(); r != nil {
			fmt.Println("Error:", r)
			saveDB()
		}
	}()
	http.HandleFunc("/", handler)
	if err := http.ListenAndServe(":8080", nil); err != nil {
		panic(err)
	}
}
