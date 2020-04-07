/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file partition_graph_in_unorder.cc
 * \brief Pass to partition a Relay graph to subgraph with unordered operators.
 */
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include "partition_graph_util.h"
namespace tvm {
namespace relay {

class GraphPartitionerInUnorder: public SubgraphPartitionor{
public:

	struct Subgraph {
	public:
		// \brief The input arguments of this subgraph.
		std::vector<std::pair<Expr, Var>> args;
		// \brief Function name.
		std::ostringstream func_name;
		// Get a new parameter or allocate an old one.
		Var GetOrAllocParam(const Expr &expr, const Type &type) {
			for (auto arg : args) {
				if (expr.same_as(arg.first))
					return arg.second;
			}
			// create a new parameter.
			std::ostringstream os;
			os << "p" << args.size();
			auto var = VarNode::make(os.str(), type);
			args.push_back( { expr, var });
			return var;
		}
	};

	/*
	 * \brief Check if the node has attrs is in attrs_.
	 * \param Relay expression.
	 * \return ture or false.
	 */
	bool HaveAttrs(const CallNode *call_node) {
		auto op = call_node->op.as<OpNode>();
		if (!op)
			return false;
		for(size_t order=0;order<op_names_.size();order++){
			LOG_IF(FATAL, !op_names_[order].as<tir::StringImmNode>())
					<<"The "<<order<<"-th op name in op_names is not defined.";
			if (op->name.compare(op_names_[order].as<tir::StringImmNode>()->value)
					== 0) {
				if (attrs_[order].operator->()) {
					LOG_IF(FATAL, op->attrs_type_index!=attrs_[order]->type_index())
																								<< "The input attrs type is not consistant with the op name, \""
																								<< op->name
																								<< "\" .";
					return AttrsComparor::CompareNonDefault(attrs_[order],
							call_node->attrs);
				} else
					return true;
			}
		}
		return false;
	}

	/*
	 * \brief Make a call function node, annotate its device type
	 * 	and cast the output data type to the raw data type.
	 * \param subgraph The Subgraph which the node belongs to.
	 * \param body The body of the function.
	 * \param raw_type The raw type of the original expression.
	 * \return The new expression.
	 */
	Expr MakeCastDTypeDeviceFunc(
			std::shared_ptr<Subgraph> &subgraph, Expr &body,
			const Type raw_type) {
		Array<Var> params;
		Array<Expr> arguments;
		for (auto pair : subgraph->args) {
			arguments.push_back(pair.first);
			params.push_back(pair.second);
		}
		auto func = FunctionNode::make(params, body, InferType(body), { });
		if (func_name_.as<tir::StringImmNode>()) {
			//Set defined function name
			func = FunctionSetAttr(func, attr::kName, func_name_);
		} else if (func_name_.as<IntImmNode>()
				&& func_name_.as<IntImmNode>()->value == 1) {
			// Auto set function name.
			func = FunctionSetAttr(func, attr::kName,
					tvm::PrimExpr(subgraph->func_name.str()));
		} else if (func_name_.as<IntImmNode>()
				&& func_name_.as<IntImmNode>()->value == 0) {
			// Do not set function name
		} else {
			LOG(WARNING)
					<< "The type of func name must be boolean or string.";
		}
		func = FunctionSetAttr(func, attr::kPrimitive, tvm::Integer(1));
		Expr new_call = CallNode::make(func, arguments, Attrs());
		if (device_type_ != 0) {
			new_call = on_device_(new_call, device_type_);
		}
		auto cast_type = raw_type.as<TensorTypeNode>()->dtype;
		new_call = Cast(new_call, cast_type);
		return new_call;
	}
	Expr VisitExpr_(const CallNode *call_node)
	final {
		Expr new_expr;
		Array<Expr> new_args;
		if (!HaveAttrs(call_node)) {
			/*
			 * If the call node's op is not in attrs_.
			 * set current subgraph to nullptr, and visit in the general way.
			 */
			current_ = nullptr;
			new_expr = ExprMutator::VisitExpr_(call_node);
		} else {
			if (current_) {
				/* If the call node belongs to a subgraph.*/
				if(visit_counter_[call_node]>1)
				{
					/*
					* if call_node is referenced by more than one operators.
					* Make a param and return.
					*/
					auto current = current_;
					current_ = nullptr;
					new_expr = ExprMutator::VisitExpr_(call_node);
					new_expr = Cast(new_expr, data_type_);
					new_expr = current->GetOrAllocParam(new_expr, InferType(new_expr));
				}else{
					/*
					* Otherwise, visit its args and get or allocate new param to subgraph for each arg
					* except of the call node with op which is in attrs_.
					*/
					for (auto arg : call_node->args) {
						Expr new_arg = this->VisitExpr(arg);
						auto is_call = new_arg.as<CallNode>();
						if (!is_call || !HaveAttrs(is_call)) {
							new_arg = Cast(new_arg, data_type_);
							if (!new_arg.as<ConstantNode>()) {
								new_arg = current_->GetOrAllocParam(new_arg,
										InferType(new_arg));
							}
						}
						new_args.push_back(new_arg);
					}
					new_expr = CallNode::make(call_node->op, new_args,
							call_node->attrs, call_node->type_args);
					current_->func_name << "_";
					current_->func_name
							<< ConvertOpName(call_node->op.as<OpNode>()->name);
				}
			} else {
				/* If the call node doesn't belong to any subgraph.
				 * visit its args and get or allocate new param to subgraph for each arg
				 * except of the call node with op which is in attrs_.
				 * Afterwards, make a function.
				 */
				current_ = std::make_shared<Subgraph>();
				auto new_subgraph = current_;
				for (auto arg : call_node->args) {
					Expr new_arg = this->VisitExpr(arg);
					auto is_call = new_arg.as<CallNode>();
					if (!is_call || (is_call && !HaveAttrs(is_call))) {
						new_arg = Cast(new_arg, data_type_);
						if (!new_arg.as<ConstantNode>()) {
							new_arg = current_->GetOrAllocParam(new_arg,
									InferType(new_arg));
						}
					}
					new_args.push_back(new_arg);
				}
				new_expr = CallNode::make(call_node->op, new_args,
						call_node->attrs, call_node->type_args);
				new_expr = MakeCastDTypeDeviceFunc(new_subgraph, new_expr,
						call_node->checked_type());
				current_->func_name << "_";
				current_->func_name
						<< ConvertOpName(call_node->op.as<OpNode>()->name);
			}
		}
		return new_expr;
	}

	Expr VisitExpr(const Expr &expr) {
		// Store the current subgraph to maintain for the next branch.
		auto subgraph = current_;
		auto it = this->memo_.find(expr);
		if (it != this->memo_.end()) {
			return it->second;
		} else {
			Expr new_expr = ExprFunctor::VisitExpr(expr);
			memo_[expr] = new_expr;
			current_ = subgraph;
			return new_expr;
		}
	}
	explicit GraphPartitionerInUnorder(const Array<PrimExpr> op_names,
			const Array<Attrs> attrs, const PrimExpr func_name,
			const int &device_type, const DataType &data_type) :
			op_names_(op_names), attrs_(attrs), func_name_(func_name), device_type_(
					device_type), data_type_(data_type) {
	}
	class RefVisitor: public ExprVisitor {
	public:
		std::unordered_map<const Object*, size_t> GetCounter(Expr expr) {
			ExprVisitor::VisitExpr(expr);
			return visit_counter_;
		}
	};
	Expr Partition(Expr expr) {

		auto visitor = RefVisitor();
		visit_counter_ = visitor.GetCounter(expr);
		return ExprMutator::Mutate(expr);
	}
	;
private:
	// The packed function to make on_device_ operator.
	const PackedFunc on_device_ = (*tvm::runtime::Registry::Get(
			"relay.op.annotation._make.on_device"));
	// The pointer point to current subgraph.
	std::shared_ptr<Subgraph> current_ { nullptr };
	// The names of the searched operator.
	const Array<PrimExpr> op_names_;
	// The attribute of the searched operator.
	const Array<Attrs> attrs_;
	// The function name.
	const PrimExpr func_name_;
	// The annotated device type for subgraphs.
	int device_type_;
	// The data type which the function computes in.
	const DataType data_type_;
	// Internal visiting counter
	std::unordered_map<const Object*, size_t> visit_counter_;

}
;

Expr PartitionGraphInUnorder(const Array<PrimExpr> op_names,
		Array<Attrs> attrs, const PrimExpr func_name, const int &device_type,
		const DataType &data_type, const Expr &expr) {
	if (op_names.size() == 0 && attrs.size() == 0) {
		LOG(FATAL) << "Op names and attrs is not set.";
		return expr;
	} else if (attrs.size() == 0) {
		auto n = make_object<ArrayNode>();
		n->data.resize(op_names.size());
		attrs = Array<Attrs>(n);
	}else if (op_names.size() > attrs.size()) {
		attrs.resize(op_names.size());
	}  else if (op_names.size() < attrs.size()) {
		LOG(WARNING) << "The length of op names must be longer than attrs.";
	} else
		;
	return GraphPartitionerInUnorder(op_names, attrs, func_name, device_type,
			data_type).Partition(expr);
}

namespace transform {
Pass PartitionGraphInUnorder(Array<PrimExpr> op_names, Array<Attrs> attrs,
		PrimExpr func_name, int device_type, DataType data_type) {
	runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
			[=](Function f, IRModule m, PassContext pc) {
				return Downcast<Function>(
						PartitionGraphInUnorder(op_names, attrs, func_name,
								device_type, data_type, f));
			};
	return CreateFunctionPass(pass_func, 1, "PartitionGraphInUnorder", {
			tir::StringImmNode::make("InferType") });
}

TVM_REGISTER_GLOBAL("relay._transform.PartitionGraphInUnorder")
		.set_body_typed(PartitionGraphInUnorder);
}  // namespace transform
}  // namespace relay
}  // namespace tvm

