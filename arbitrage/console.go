package main

import "os"
import "bufio"
import "fmt"
import "bytes"
import "encoding/json"

import "golang.org/x/net/websocket"
import "github.com/gizak/termui"

const (
	ACCESS_KEY = "********-****-****-****-************"
	SECRET_KEY = "********-****-****-****-************"
	SERVER     = "ws://api.chbtc.com:8888/websocket"
	N          = 15
)

type Conn struct {
	websocket.Conn
}

func Dial() (*Conn, error) {
	ws, err := websocket.Dial(SERVER, "", "http://localhost/")
	return &Conn{*ws}, err
}
func (ws *Conn) Listen() {
}

func (ws *Conn) Order() {
}

var CURRENCIES = map[string]string{
	"btc_cny_depth": "btccny",
	"eth_cny_depth": "ethcny",
	"eth_btc_depth": "ethbtc",
}

var CHANNELS = map[string]string{
	"btccny": "btc_cny_depth",
	"ethcny": "eth_cny_depth",
	"ethbtc": "eth_btc_depth",
}

type Depth struct {
	Asks    [][2]float64 `json:asks`
	Bids    [][2]float64 `json:bids`
	No      int          `json:no`
	Channel string       `json:channel`
}

func Depths(currencies []string) (map[string]chan Depth, error) {
	depths := make(map[string]chan Depth)
	for _, currency := range currencies {
		depths[currency] = make(chan Depth, 10)
	}

	if ws, err := Dial(); err == nil {
		s := map[string]string{"event": "addChannel"}
		for _, currency := range currencies {
			if channel, ok := CHANNELS[currency]; ok {
				s["channel"] = channel
				if data, err := json.Marshal(s); err == nil {
					ws.Write(data)
				} else {
					panic(err)
				}
			}
		}
		go func() {
			var err error
			var n int
			var buffer bytes.Buffer
			msg := make([]byte, 8*1024)
			nos := make(map[string]int)

			for ; err == nil; n, err = ws.Read(msg) {
				if n == 0 {
					continue
				}
				buffer.Write(msg[:n])
				if msg[n-1] != 125 { // 125 is }
					continue
				}
				if n, err = buffer.Read(msg); err != nil {
					panic(err)
				}
				var depth Depth
				if err = json.Unmarshal(msg[:n], &depth); err != nil {
					panic(err)
				}
				if currency, ok := CURRENCIES[depth.Channel]; ok {
					if depth.No > nos[currency] {
						nos[currency] = depth.No
						depths[currency] <- depth
					}
				}
			}
		}()
		return depths, nil
	} else {
		return depths, err
	}
}

func parseOrders(orders [][2]float64, n int, f string) (items []string) {
	var s, e int
	if n < 0 {
		e = len(orders)
		s = e + n
	} else {
		e = n
		s = 0
	}
	for ; s < e; s++ {
		items = append(items, fmt.Sprintf(f, orders[s][0], orders[s][1]))
	}
	return
}

func main() {
	currencies := []string{"btccny", "ethcny", "ethbtc"}
	fmts := map[string]string{
		"btccny": "%8.2f %8.3f",
		"ethcny": "%6.2f %10.3f",
		"ethbtc": "%8.6f %10.3f"}

	charts := make(map[string]map[string]*termui.List)
	consoles := make(map[string]*termui.List)

	if err := termui.Init(); err == nil {
		consoles_row := termui.NewRow()
		bot := termui.NewList()
		bot.BorderLabel = "bot logs"
		bot.Height = 7
		man := termui.NewList()
		man.BorderLabel = "console"
		man.Height = 7
		consoles_row.Cols = []*termui.Row{
			termui.NewCol(6, 0, bot), termui.NewCol(6, 0, man)}
		consoles = map[string]*termui.List{"bot": bot, "man": man}

		buys := termui.NewRow()
		sells := termui.NewRow()
		for _, currency := range currencies {
			buy := termui.NewList()
			buy.BorderLabelFg = termui.ColorYellow
			buy.ItemFgColor = termui.ColorRed
			buy.Height = N + 2

			sell := termui.NewList()
			sell.BorderLabel = currency
			sell.BorderLabelFg = termui.ColorYellow
			sell.ItemFgColor = termui.ColorGreen
			sell.Height = N + 2

			buys.Cols = append(buys.Cols, termui.NewCol(4, 0, buy))
			sells.Cols = append(sells.Cols, termui.NewCol(4, 0, sell))

			charts[currency] = map[string]*termui.List{"buy": buy, "sell": sell}
		}

		termui.Body.AddRows(sells, buys, consoles_row)
		termui.Body.Align()
		termui.Render(termui.Body)
	} else {
		panic(err)
	}

	if depths, err := Depths(currencies); err == nil {
		go func() {
			var currency, f string
			var depth Depth
			for {
				select {
				case depth = <-depths["btccny"]:
					currency = "btccny"
				case depth = <-depths["ethcny"]:
					currency = "ethcny"
				case depth = <-depths["ethbtc"]:
					currency = "ethbtc"
				}
				f = fmts[currency]
				charts[currency]["buy"].Items = parseOrders(depth.Bids, N, f)
				charts[currency]["sell"].Items = parseOrders(depth.Asks, -N, f)
				termui.Render(termui.Body)
			}
		}()
	} else {
		panic(err)
	}

	termui.Handle("/sys/kbd/C-c", func(termui.Event) { termui.StopLoop() })
	termui.Handle("/sys/kbd/:", func(termui.Event) {
		scanner := bufio.NewScanner(os.Stdin)
		scanner.Scan()
		consoles["man"].Items = []string{scanner.Text()}
		termui.Render(termui.Body)
	})
	termui.Loop()
}
