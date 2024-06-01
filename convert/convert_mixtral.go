package convert

import (
	"encoding/binary"
	"fmt"
	"io"
	"slices"
	"strings"

	"github.com/pdevine/tensor"

	"github.com/ollama/ollama/llm"
)

type mixtral struct {
	llama
	NumLocalExperts    uint32 `json:"num_local_experts"`
	NumExpertsPerToken uint32 `json:"num_experts_per_tok"`
}

func (p *mixtral) KV(t *Tokenizer) map[string]any {
	kv := p.llama.KV(t)

	if p.NumLocalExperts > 0 {
		kv["llama.expert_count"] = p.NumLocalExperts
	}

	if p.NumExpertsPerToken > 0 {
		kv["llama.expert_used_count"] = p.NumExpertsPerToken
	}

	return kv
}

func (p *mixtral) Tensors(ts []Tensor) []*llm.Tensor {
	oldnew := []string{
		"model.layers", "blk",
		"w1", "ffn_gate_exps",
		"w2", "ffn_down_exps",
		"w3", "ffn_up_exps",
	}

	for i := range p.NumLocalExperts {
		oldnew = append(oldnew, fmt.Sprintf(".block_sparse_moe.experts.%d.", i), ".")
	}

	namer := strings.NewReplacer(oldnew...)

	experts := make(map[string]expert)
	ts = slices.DeleteFunc(ts, func(t Tensor) bool {
		if !strings.Contains(t.Name(), ".block_sparse_moe.experts.") {
			return false
		}

		name := namer.Replace(t.Name())
		experts[name] = append(experts[name], t)
		return true
	})

	var out []*llm.Tensor
	for n, e := range experts {
		out = append(out, &llm.Tensor{
			Name:     n,
			Kind:     e.Kind(),
			Shape:    e.Shape(),
			WriterTo: e,
		})
	}

	return append(out, p.llama.Tensors(ts)...)
}

type expert []Tensor

func (e expert) Kind() uint32 {
	return e[0].Kind()
}

func (e expert) Shape() []uint64 {
	return append([]uint64{uint64(len(e))}, e[0].Shape()...)
}

func (e expert) WriteTo(w io.Writer) (int64, error) {
	var dims []int
	for _, dim := range e[0].Shape() {
		dims = append(dims, int(dim))
	}

	var stack []tensor.Tensor
	for _, t := range e {
		var b uint16s
		if _, err := t.WriteTo(&b); err != nil {
			return 0, err
		}

		stack = append(stack, tensor.New(tensor.WithShape(dims...), tensor.WithBacking(b.Bits())))
	}

	stacked, err := tensor.Stack(0, stack[0], stack[1:]...)
	if err != nil {
		return 0, err
	}

	err = binary.Write(w, binary.LittleEndian, stacked.Data())
	return 0, err
}

type uint16s struct {
	b []uint16
}

func (u *uint16s) Write(p []byte) (int, error) {
	if len(p)%2 != 0 {
		return 0, fmt.Errorf("odd length")
	}

	var n int
	for n = 0; n < len(p); n += 2 {
		u.b = append(u.b, binary.LittleEndian.Uint16(p[n:]))
	}

	return n, nil
}

func (u *uint16s) Bits() []uint16 {
	return u.b
}
