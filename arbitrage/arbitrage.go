package main

import (
	"crypto/hmac"
	"crypto/md5"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"net/http"
	"strings"
	"time"
)

import "golang.org/x/net/websocket"

const (
	ACCESS_KEY = "f*******-****-****-****-************"
	WS_SERVER  = "ws://api.chbtc.com:8888/websocket"
	SERVER     = "https://trade.chbtc.com/api/%s?%s&reqTime=%d000&sign=%s"
	BUY        = 1
	SELL       = 0
)

var SECRET_KEY = func() []byte {
	sha := sha1.New()
	io.WriteString(sha, "********-****-****-****-************")
	sum := sha.Sum(nil)
	secret := make([]byte, 2*len(sum))
	hex.Encode(secret, sum)
	return secret
}()

func Sig(msg []byte) string {
	hm := hmac.New(md5.New, SECRET_KEY)
	hm.Write(msg)
	return hex.EncodeToString(hm.Sum(nil))
}

var COUNTER uint64 = 0

func uuid() string {
	COUNTER += 1
	return fmt.Sprintf("%x-%x", time.Now().Unix(), COUNTER)
}

var DEPTH_CHANNELS = map[string]string{
	"btc_cny_depth": "btc_cny_order",
	"eth_cny_depth": "eth_cny_order",
	"eth_btc_depth": "eth_btc_order",
}

type Depth struct {
	Asks    [][2]float64 `json:asks`
	Bids    [][2]float64 `json:bids`
	No      int          `json:no`
	Channel string       `json:channel`
}

func Depths() (chan Depth, error) {
	depths := make(chan Depth, 1000)
	if ws, err := websocket.Dial(WS_SERVER, "", "http://localhost/"); err == nil {
		s := map[string]string{"event": "addChannel"}
		for channel, _ := range DEPTH_CHANNELS {
			s["channel"] = channel
			if err := websocket.JSON.Send(ws, s); err != nil {
				return depths, err
			}
		}
		go func() {
			nos := make(map[string]int)

			for {
				var depth Depth
				if err := websocket.JSON.Receive(ws, &depth); err != nil {
					break
				}
				if _, ok := DEPTH_CHANNELS[depth.Channel]; ok && depth.No > nos[depth.Channel] {
					nos[depth.Channel] = depth.No
					depths <- depth
				}
			}
		}()
		return depths, nil
	} else {
		return depths, err
	}
}

type Order struct {
	Event     string  `json:"event"`
	Channel   string  `json:"channel"`
	Accesskey string  `json:"accesskey"`
	TradeType int     `json:"tradeType"`
	Price     float64 `json:"price"`
	Amount    float64 `json:"amount"`
	No        string  `json:"no,omitempty"`
	Sign      string  `json:"sign,omitempty"`
}

type OrderResp struct {
	Success bool              `json:success`
	Code    int               `json:code`
	Data    map[string]uint64 `json:data`
	Channel string            `json:channel`
	Message string            `json:message`
	No      string            `json:no`
}

type Porter struct {
	// for order use only
	Ws      *websocket.Conn
	Depths  map[string]Depth
	Resps   chan OrderResp
	Balance map[string]float64
	Frozen  map[string]float64
}

func NewPorter() (*Porter, error) {
	if ws, err := websocket.Dial(WS_SERVER, "", "http://localhost/"); err == nil {
		porter := &Porter{
			ws,
			make(map[string]Depth),
			make(chan OrderResp, 1000),
			make(map[string]float64),
			make(map[string]float64)}
		go func() {
			for {
				var resp OrderResp
				if err := websocket.JSON.Receive(ws, &resp); err == nil {
					porter.Resps <- resp
				} else {
					fmt.Println(err)
				}
			}
		}()
		return porter, porter.SyncAccount()
	} else {
		return &Porter{}, nil
	}
}

func (self *Porter) update(depth Depth) {
	if channel, ok := DEPTH_CHANNELS[depth.Channel]; ok {
		self.Depths[channel] = depth
	}
}

var ARBITRAGES = [][]*Order{
	// buy eth, sell ethbtc, sell btc
	[]*Order{
		&Order{Event: "addChannel", Channel: "eth_cny_order", Accesskey: ACCESS_KEY, TradeType: BUY},
		&Order{Event: "addChannel", Channel: "btc_cny_order", Accesskey: ACCESS_KEY, TradeType: SELL},
		&Order{Event: "addChannel", Channel: "eth_btc_order", Accesskey: ACCESS_KEY, TradeType: SELL}},
	// buy btc, buy ethbtc, sell eth
	[]*Order{
		&Order{Event: "addChannel", Channel: "eth_cny_order", Accesskey: ACCESS_KEY, TradeType: SELL},
		&Order{Event: "addChannel", Channel: "btc_cny_order", Accesskey: ACCESS_KEY, TradeType: BUY},
		&Order{Event: "addChannel", Channel: "eth_btc_order", Accesskey: ACCESS_KEY, TradeType: BUY}},
}

func (self *Porter) Analyse(benefit float64) ([]*Order, bool) {
	for _, orders := range ARBITRAGES {
		for _, order := range orders {
			if depth, ok := self.Depths[order.Channel]; ok {
				var _order [2]float64
				p := strings.Split(order.Channel, "_")[order.TradeType]
				if order.TradeType == BUY {
					_order = depth.Asks[len(depth.Asks)-1]
					order.Amount = math.Min(self.Balance[p]/_order[0], _order[1])
				} else if order.TradeType == SELL {
					_order = depth.Bids[0]
					order.Amount = math.Min(self.Balance[p], _order[1])
				}
				order.Price = _order[0]
			} else {
				return orders, ok
			}
		}
		diff := orders[1].Price * orders[2].Price / orders[0].Price
		if orders[0].TradeType == BUY {
			diff -= 1
		} else if orders[0].TradeType == SELL {
			diff = 1 - diff
		}
		if 3*diff > benefit {
			ea := math.Min(orders[0].Amount, orders[2].Amount)
			ba := math.Min(ea*orders[2].Price, orders[1].Amount)
			if diff < benefit {
				ba = ba * diff / benefit
			}
			ba = math.Floor(1000*ba) / 1000
			ea = math.Floor(100*ba/orders[2].Price) / 100
			if ba > 0 && ea > 0 {
				orders[0].Amount = ea
				orders[1].Amount = ba
				orders[2].Amount = ea
				fmt.Println(orders[0].Price - orders[1].Price*orders[2].Price)
				return orders, true
			}
		}
	}
	return []*Order{}, false
}

func (self *Porter) Trade(orders []*Order, timeout time.Duration) error {
	_orders := make(map[string]*Order)
	for _, order := range orders {
		order.No = uuid()
		order.Sign = ""
		if msg, err := json.Marshal(*order); err == nil {
			order.Sign = Sig(msg)
		} else {
			return err
		}
		_orders[order.No] = order
		if err := websocket.JSON.Send(self.Ws, *order); err != nil {
			return err
		}
		currency := strings.Split(order.Channel, "_")[order.TradeType]
		v := order.Amount
		if order.TradeType == BUY {
			v = v * order.Price
		}
		self.Frozen[currency] += v
		self.Balance[currency] -= v
	}
	for len(_orders) > 0 {
		select {
		case <-time.After(timeout):
			return errors.New("trade timeout")
		case resp := <-self.Resps:
			if order, ok := _orders[resp.No]; ok {
				if !resp.Success {
					fmt.Println(resp)
					fmt.Println(*order)
				}
				delete(_orders, resp.No)
			}
		}
	}
	return nil
}

type Account struct {
	Data struct {
		Frozen struct {
			CNY struct {
				Amount float64 `json:"amount"`
			} `json:"cny"`
			BTC struct {
				Amount float64 `json:"amount"`
			} `json:"btc"`
			ETH struct {
				Amount float64 `json:"amount"`
			} `json:"eth"`
		} `json:"frozen"`
		Balance struct {
			CNY struct {
				Amount float64 `json:"amount"`
			} `json:"cny"`
			BTC struct {
				Amount float64 `json:"amount"`
			} `json:"btc"`
			ETH struct {
				Amount float64 `json:"amount"`
			} `json:"eth"`
		} `json:"balance"`
	} `json:"result"`
}

func (self *Porter) checkBalance() string {
	for _, currency := range [3]string{"cny", "btc", "eth"} {
		if self.Frozen[currency] > self.Balance[currency] {
			return currency
		}
	}
	return ""
}

func (self *Porter) SyncAccount() error {
	q := "method=getAccountInfo&accesskey=" + ACCESS_KEY
	u := fmt.Sprintf(SERVER, "getAccountInfo", q, time.Now().Unix(), Sig([]byte(q)))
	if resp, err := http.Get(u); err == nil {
		if resp.StatusCode == 200 {
			defer resp.Body.Close()
			if content, err := ioutil.ReadAll(resp.Body); err == nil {
				var resp Account
				if err := json.Unmarshal(content, &resp); err == nil {
					self.Balance["cny"] = resp.Data.Balance.CNY.Amount
					self.Balance["btc"] = resp.Data.Balance.BTC.Amount
					self.Balance["eth"] = resp.Data.Balance.ETH.Amount
					self.Frozen["cny"] = resp.Data.Frozen.CNY.Amount
					self.Frozen["btc"] = resp.Data.Frozen.BTC.Amount
					self.Frozen["eth"] = resp.Data.Frozen.ETH.Amount
				}
			}
		} else {
			return errors.New("sync account failed")
		}
	} else {
		return err
	}
	return nil
}

type Query struct {
	Currency  string `json:"currency"`
	Id        string `json:"id"`
	TradeType int    `json:"type"`
}

func (self *Porter) CancelOrders(currency string) {
	var queries []Query
	switch currency {
	case "cny":
		queries = []Query{Query{"btc_cny", "", BUY}, Query{"eth_cny", "", BUY}}
	case "btc":
		queries = []Query{Query{"btc_cny", "", SELL}, Query{"eth_btc", "", BUY}}
	case "eth":
		queries = []Query{Query{"eth_cny", "", SELL}, Query{"eth_btc", "", SELL}}
	default:
		panic(currency)
	}
	for _, query := range queries {
		q := fmt.Sprintf("method=getUnfinishedOrdersIgnoreTradeType&accesskey=%s&currency=%s&pageIndex=1&pageSize=100", ACCESS_KEY, query.Currency)
		u := fmt.Sprintf(SERVER, "getUnfinishedOrdersIgnoreTradeType", q, time.Now().Unix(), Sig([]byte(q)))
		if resp, err := http.Get(u); err == nil {
			if resp.StatusCode == 200 {
				defer resp.Body.Close()
				if content, err := ioutil.ReadAll(resp.Body); err == nil {
					var resps []Query
					if err := json.Unmarshal(content, &resps); err == nil {
						for _, resp := range resps {
							if resp.Currency == query.Currency && resp.TradeType == query.TradeType {
								q := fmt.Sprintf("method=cancelOrder&accesskey=%s&id=%s&currency=%s", ACCESS_KEY, resp.Id, resp.Currency)
								u := fmt.Sprintf(SERVER, "cancelOrder", q, time.Now().Unix(), Sig([]byte(q)))
								http.Get(u)
							}
						}
					}
				}
			} else {
				fmt.Println(resp.StatusCode)
			}
		} else {
			fmt.Println(err)
		}
	}
}

func main() {
	benefit := flag.Float64("benefit", 0.0005, "benefit rate")
	flag.Parse()
	if porter, err := NewPorter(); err == nil {
		if depths, err := Depths(); err == nil {
			for {
				select {
				case depth := <-depths:
					porter.update(depth)
				default:
					if orders, ok := porter.Analyse(*benefit); ok {
						if err := porter.Trade(orders, time.Second); err == nil {
							if porter.checkBalance() != "" {
								for {
									time.Sleep(time.Second * 2)
									if err := porter.SyncAccount(); err == nil {
										currency := porter.checkBalance()
										if currency == "" {
											break
										}
										porter.CancelOrders(currency)
									} else {
										fmt.Println(err)
									}
								}
							}
						} else {
							fmt.Println(err)
						}
					}
					porter.update(<-depths)
				}
			}
		} else {
			panic(err)
		}
	} else {
		panic(err)
	}
}
